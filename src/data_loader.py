import rasterio
from rasterio.errors import NotGeoreferencedWarning
import numpy as np
from enum import Enum
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import json
import random
from typing import Union, Optional, Callable, Tuple, List, Literal
import warnings

def load_sar_image(path: Union[str, Path]) -> np.ndarray:
    """Load SAR image as single-band array.

    - For PNG/JPEG, use PIL (no georeference expected, avoids Rasterio warnings).
    - For GeoTIFF or others, use Rasterio and suppress NotGeoreferencedWarning.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {'.png', '.jpg', '.jpeg'}:
        return np.array(Image.open(p).convert('L'))
    # Fall back to Rasterio for formats like GeoTIFF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(p) as src:
            image = src.read(1)
    return image

def to_pil_grayscale(image: np.ndarray) -> Image.Image:
    """Convert a single-channel numpy array to PIL grayscale image."""
    # Clip to valid byte range and convert to uint8 for visualization/consistency
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode='L')

class SplitType(Enum):
    """Enumeration for dataset split types"""
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class Sentinel(Dataset):
    def __init__(self,
                 root_dir: str = r"D:\COLLEGE\SAR\2.7 Gb V_2\v_2",
                 split_type: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 split_mode: Literal['random', 'split'] = 'random',
                 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 split_file: Optional[Union[str, Path]] = None,
                 seed: int = 42):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")
        
        self.split_type = SplitType(split_type) if split_type else None
        # Transforms apply to both SAR and RGB; default normalizes to [0,1] float32
        self.transform = transform if transform else v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.all_image_pairs = self._collect_images()
        
        if split_type:
            if split_mode == 'split' and split_file:
                self.image_pairs = self._apply_predefined_split(split_file)
            elif split_mode == 'random':
                self.image_pairs = self._apply_random_split(split_ratio, seed)
            else:
                raise ValueError("Invalid split configuration")
        else:
            self.image_pairs = self.all_image_pairs

        print(f'Total image pairs found: {len(self)}')

    def _collect_images(self) -> List[Tuple[Path, Path]]:
        image_pairs = []
        for category in self.root_dir.iterdir():
            if not category.is_dir():
                continue

            s1_path = category / 's1'
            s2_path = category / 's2'
            
            if not (s1_path.is_dir() and s2_path.is_dir()):
                continue

            for s1_file in s1_path.glob('*.png'):
                s2_filename = list(s1_file.name.split('_'))
                s2_filename[2] = 's2'
                s2_file = s2_path / '_'.join(s2_filename)

                if not s2_file.exists():
                    continue

                image_pairs.append((s1_file, s2_file))
        
        return image_pairs

    def _apply_random_split(self, split_ratio: Tuple[float, float, float], seed: int) -> List[Tuple[Path, Path]]:
        train_ratio, val_ratio, test_ratio = split_ratio
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("split_ratio must sum to 1.0")
        rng = random.Random(seed)
        pairs = list(self.all_image_pairs)
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if self.split_type == SplitType.TRAIN:
            return pairs[:n_train]
        if self.split_type == SplitType.VAL:
            return pairs[n_train:n_train + n_val]
        if self.split_type == SplitType.TEST:
            return pairs[n_train + n_val:]
        return pairs

    def _apply_predefined_split(self, split_file: Union[str, Path]) -> List[Tuple[Path, Path]]:
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        data = json.loads(split_path.read_text())
        key = self.split_type.value if self.split_type else 'all'
        selected_rel = set(data.get(key, []))
        selected: List[Tuple[Path, Path]] = []
        for s1, s2 in self.all_image_pairs:
            rel = str(s1.relative_to(self.root_dir))
            if rel in selected_rel:
                selected.append((s1, s2))
        return selected

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s1_path, s2_path = self.image_pairs[idx]
        # Load SAR as 1-channel grayscale
        sar_np = load_sar_image(s1_path)
        sar_pil = to_pil_grayscale(sar_np)
        # Load RGB target
        s2_image = Image.open(s2_path).convert('RGB')

        sar_tensor = self.transform(sar_pil)  # shape [1, H, W]
        rgb_tensor = self.transform(s2_image)  # shape [3, H, W]

        # Optional normalization to [-1,1] if model expects it: handled in training transforms
        return sar_tensor, rgb_tensor