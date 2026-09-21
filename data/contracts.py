from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    group: str
    split: str = ""


@dataclass(frozen=True)
class TileRecord:
    source: SampleRecord
    # Pixel coordinates in the original scene; edge windows may extend past it.
    window: tuple[int, int, int, int]  # top, left, height, width
    original_size: tuple[int, int]
