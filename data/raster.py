"""Read ordinary images and TIFF rasters without silently discarding bands."""
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def read_array(path, rgb_palette=False):
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = series.axes
            array = series.asarray()
        if array.ndim == 3 and axes in {"SYX", "CYX"}:
            array = array.transpose(1, 2, 0)
        elif array.ndim == 3 and axes not in {"YXS", "YXC"}:
            raise ValueError(f"Unsupported TIFF axes {axes} in {path}; expected a single YX, YXS or SYX raster")
        if array.ndim not in {2, 3}:
            raise ValueError(f"Unsupported TIFF shape {array.shape}: {path}")
        return array
    with Image.open(path) as image:
        if rgb_palette and image.mode == "P":
            image = image.convert("RGB")
        return np.array(image)


def raster_size(path):
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes, shape = series.axes, series.shape
            if axes not in {"YX", "YXS", "YXC", "SYX", "CYX"}:
                raise ValueError(f"Unsupported TIFF axes {axes}: {path}")
            return shape[axes.index("Y")], shape[axes.index("X")]
    with Image.open(path) as image:
        return image.height, image.width


def read_image(path, cfg):
    array = read_array(path, rgb_palette=True)
    if array.ndim == 2:
        array = array[..., None]
    bands = cfg.get("bands")
    if bands is not None:
        if max(bands) >= array.shape[-1]:
            raise ValueError(f"Requested bands {bands}, but image has {array.shape[-1]} channels: {path}")
        array = array[..., bands]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {array.shape[-1]}: {path}. Configure images.bands explicitly.")
    scale = cfg.get("scale", 255.0)
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > scale:
        raise ValueError(f"Image values must be finite in [0, images.scale={scale}]: {path}")
    # Preserve integer storage in the per-worker scene cache; normalize each tile.
    return array


def read_mask(path, labels):
    encoding = labels.get("encoding", "index")
    array = read_array(path, rgb_palette=encoding == "rgb")
    ignore = labels.get("ignore_index", 255)
    if encoding == "rgb":
        if array.ndim != 3 or array.shape[-1] != 3 or not np.issubdtype(array.dtype, np.integer) or array.min() < 0 or array.max() > 255:
            raise ValueError(f"Expected an integer RGB mask in [0,255]: {path}")
        keys = array.astype(np.int64)
        keys = (keys[..., 0] << 16) | (keys[..., 1] << 8) | keys[..., 2]
        mapping = {}
        for color, target in labels["colors"].items():
            red, green, blue = (int(v.strip()) for v in str(color).split(","))
            mapping[(red << 16) | (green << 8) | blue] = target
    else:
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Expected a single-channel integer mask: {path}")
        keys = array
        if encoding == "binary":
            result = (array > labels.get("threshold", 127)).astype(np.int64)
            for value in labels.get("ignore_values", []):
                result[array == value] = ignore
            return result
        mapping = labels["mapping"]
    unknown = set(np.unique(keys).tolist()) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped mask values {sorted(unknown)[:12]} in {path}; map them explicitly, including void labels")
    result = np.full(keys.shape, ignore, dtype=np.int64)
    for source, target in mapping.items():
        result[keys == source] = target
    return result
