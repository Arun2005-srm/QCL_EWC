import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .contracts import TileRecord
from .raster import raster_size, read_image, read_mask


def _starts(length, tile, stride):
    # Cover every pixel, including a final edge window, without redundant tails.
    starts = [0]
    while starts[-1] + tile < length:
        starts.append(starts[-1] + stride)
    return starts


def make_tiles(records, cfg):
    tiles = []
    tiling = cfg.get("tiling", {})
    for record in records:
        height, width = raster_size(record.image_path)
        if raster_size(record.mask_path) != (height, width):
            raise ValueError(f"Image and mask dimensions differ: {record.sample_id}")
        if tiling.get("enabled", False):
            th, tw = tiling.get("size", [512, 512])
            sh, sw = tiling.get("stride", [th, tw])
            for top in _starts(height, th, sh):
                for left in _starts(width, tw, sw):
                    tiles.append(TileRecord(record, (top, left, th, tw), (height, width)))
        else:
            tiles.append(TileRecord(record, (0, 0, height, width), (height, width)))
    return tiles


class SegmentationDataset(Dataset):
    """Dataset-blind RGB float images, long masks, and source spatial metadata.

    One decoded scene is cached per worker. TIFF reads decode one full scene;
    use pre-tiled inputs with scene groups for rasters too large for host RAM.
    """

    def __init__(self, records, cfg, training=False):
        self.cfg = cfg
        self.training = training
        self.tiles = make_tiles(records, cfg)
        self._cached_path = None
        self._cache = None

    def __len__(self):
        return len(self.tiles)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cached_path=None, _cache=None)
        return state

    def __getitem__(self, index):
        tile = self.tiles[index]
        source = tile.source
        key = (source.image_path, source.mask_path)
        if key != self._cached_path:
            image = read_image(source.image_path, self.cfg.get("images", {}))
            mask = read_mask(source.mask_path, self.cfg["labels"])
            if image.shape[:2] != mask.shape or image.shape[:2] != tile.original_size:
                raise ValueError(f"Image/mask dimensions changed or disagree: {source.sample_id}")
            self._cache, self._cached_path = (image, mask), key
        image, mask = self._cache
        top, left, height, width = tile.window
        crop_image = image[top:top + height, left:left + width]
        crop_mask = mask[top:top + height, left:left + width]
        valid_h, valid_w = crop_mask.shape
        image_tensor = torch.from_numpy(np.array(crop_image, dtype=np.float32, copy=True)).permute(2, 0, 1)
        image_tensor = image_tensor / self.cfg.get("images", {}).get("scale", 255.0)
        mask_tensor = torch.from_numpy(np.array(crop_mask, dtype=np.int64, copy=True))
        padding = (0, width - valid_w, 0, height - valid_h)
        image_tensor = F.pad(image_tensor, padding)
        mask_tensor = F.pad(mask_tensor, padding, value=self.cfg["labels"].get("ignore_index", 255))
        transform = self.cfg.get("transforms", {})
        size = transform.get("image_size", [512, 512])
        image_tensor = F.interpolate(image_tensor[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0]
        mask_tensor = F.interpolate(mask_tensor[None, None].double(), size=size, mode="nearest")[0, 0].long()
        augmentation = transform.get("augmentation", {})
        if self.training:
            for setting, dim in (("horizontal_flip", -1), ("vertical_flip", -2)):
                if augmentation.get(setting, False) and torch.rand(()) < .5:
                    image_tensor = image_tensor.flip(dim)
                    mask_tensor = mask_tensor.flip(dim)
        normalization = transform.get("normalization")
        if normalization:
            mean = torch.tensor(normalization["mean"], dtype=torch.float32)[:, None, None]
            std = torch.tensor(normalization["std"], dtype=torch.float32)[:, None, None]
            image_tensor = (image_tensor - mean) / std
        return {
            "image": image_tensor.contiguous(), "mask": mask_tensor.contiguous(),
            "id": f"{source.sample_id}@{top},{left},{height},{width}",
            "scene_id": source.sample_id, "group": source.group, "dataset_id": self.cfg["name"],
            "window": torch.tensor(tile.window), "original_size": torch.tensor(tile.original_size),
            "valid_size": torch.tensor([valid_h, valid_w]),
        }
