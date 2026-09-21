from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .alignment import SpatialAlignment


class UniversalSAM(nn.Module):
    """Frozen SAM with trainable shared alignment and shared class prompts.

    Expects RGB [0,1] and returns class logits at the input tile resolution.
    Class prompts are learned embeddings, not ground-truth-derived prompts.
    """

    def __init__(self, classes, checkpoint=None, *, kind="quantum", qubits=4, depth=2, grid=4, sam=None):
        super().__init__()
        if sam is None:
            if checkpoint is None or not Path(checkpoint).is_file():
                raise FileNotFoundError("Supply the official SAM ViT-B checkpoint with --sam-checkpoint")
            from segment_anything import sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=None)
            sam.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.sam = sam.eval()
        self.sam.requires_grad_(False)
        self.alignment = SpatialAlignment(qubits=qubits, depth=depth, grid=grid, kind=kind)
        self.class_prompts = nn.Parameter(torch.randn(classes, 1, 256) * .02)

    def train(self, mode=True):
        super().train(mode)
        self.sam.eval()
        return self

    def forward(self, images):
        height, width = images.shape[-2:]
        target = self.sam.image_encoder.img_size
        scale = target / max(height, width)
        resized_size = (int(height * scale + .5), int(width * scale + .5))
        # Use component calls, since the public SAM forward is no_grad and
        # thresholds masks, both of which are inappropriate for training here.
        with torch.no_grad():
            resized = F.interpolate(images * 255, resized_size, mode="bilinear", align_corners=False, antialias=True)
            prepared = torch.stack([self.sam.preprocess(image) for image in resized])
            embeddings = self.sam.image_encoder(prepared)
            _, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
            position = self.sam.prompt_encoder.get_dense_pe()
        embeddings = self.alignment(embeddings)
        outputs = []
        for embedding in embeddings:
            logits, _ = self.sam.mask_decoder(
                image_embeddings=embedding[None], image_pe=position,
                sparse_prompt_embeddings=self.class_prompts,
                dense_prompt_embeddings=dense.expand(len(self.class_prompts), -1, -1, -1),
                multimask_output=False,
            )
            logits = self.sam.postprocess_masks(logits, resized_size, (height, width))
            outputs.append(logits[:, 0])
        return torch.stack(outputs)

    def trainable_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters() if value.requires_grad}

    def load_trainable_state(self, state):
        expected = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if set(state) != set(expected):
            raise ValueError("Checkpoint trainable parameters do not match the configured model")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(parameter))
