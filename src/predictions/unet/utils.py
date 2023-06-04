from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from predictions.unet.dataset import ImageMaskDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = ImageMaskDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ImageMaskDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(loader: DataLoader,
                             model: nn.Module,
                             output_folder: Path,
                             device: str = "cuda" if torch.cuda.is_available() else "cpu"
                             ):
    model.eval()
    for idx, (image_tensor, mask_tensor, image_name, mask_name) in tqdm(enumerate(loader),
                                                                        desc="Saving predictions images",
                                                                        total=len(loader)):
        image_tensor = image_tensor.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(image_tensor))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{output_folder}/{'-'.join(mask_name)}"
        )

    model.train()
