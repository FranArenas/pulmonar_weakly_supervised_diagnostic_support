from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from predictions.unet.dataset import ImageMaskDataset
from predictions.unet.unet import Unet
from predictions.unet.utils import save_predictions_as_imgs

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    dataset = ImageMaskDataset(
        image_dir=Path("../../../data/unet/test/lung"),
        mask_dir=Path("../../../data/unet/test/mask"),
        transform=transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True
    )

    model = Unet()
    checkpoint = Path("../../../data/unet/checkpoint/unet_checkpoint_v1.pth.tar").resolve()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device))["state_dict"])

    output_folder = Path("../../../data/unet/output")
    output_folder.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_predictions_as_imgs(data_loader, model, output_folder, device)
