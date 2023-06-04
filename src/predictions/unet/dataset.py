from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageMaskDataset(Dataset):
    def __init__(self,
                 image_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 transform=None):
        self.images: List[Path] = [*Path(image_dir.resolve()).rglob("*.png")]
        self.masks: List[Path] = [*Path(mask_dir.resolve()).rglob("*.png")]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img_path: Path = self.images[index]
        mask_path = [*filter(lambda x: img_path.name in str(x), self.masks)]
        assert len(mask_path) == 1, mask_path
        mask_path = mask_path[0]
        image = np.array(Image.open(str(img_path)).convert("RGB"))
        mask = np.array(Image.open(str(mask_path)).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, img_path.name, mask_path.name
