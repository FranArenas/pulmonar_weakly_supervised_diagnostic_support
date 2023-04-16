from pathlib import Path

import PIL.Image
import cv2
import torch
from tqdm import tqdm

from classification.mask_generator import MaskGenerator
from classification.predict import predict
from classification.resnet50 import Resnet50
from classification.sidu.mask_generator import SiduMaskGenerator


def generate_masks(images_path: Path,
                   masks_path: Path,
                   model: Resnet50,
                   mask_generator: MaskGenerator,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    labels_paths = [*images_path.glob("*")]

    for label_path in labels_paths:
        output_path = masks_path / label_path.name
        output_path.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm([*label_path.glob("*")], desc=f"Generating {label_path.name} masks"):

            prediction = predict(image_path, model, device=device)

            mask_path = \
                Path(f"{output_path}/"
                     f"{'correct' if prediction[label_path.name] >= 0.5 else 'wrong'}/"
                     f"covid:{prediction['covid']:.3f}-"
                     f"nocovid:{prediction['nocovid']:.3f}-"
                     f"{image_path.stem + image_path.suffix}").resolve()

            mask_path.parent.mkdir(parents=True, exist_ok=True)

            mask = mask_generator.generate_mask(PIL.Image.open(image_path))

            if not cv2.imwrite(str(mask_path), mask * 255):
                raise RuntimeError("Image was not written")


if __name__ == "__main__":
    weights = Path(
        "../../data/weights/desc=test_acc=0.7905405759811401_test_loss=0.5681480640998563_Reduced 2048-1024-512_Resnet50learning_rate=0.0001_weight_decay=5e-05_layer_act=0_nepochs=42_7e7c1b32-210f-4c70-a368-e3c77eaf.pt")
    model = Resnet50("cpu")
    model.load_weights(weights)

    images_path = Path("../../data/output/zero_padding_reduced/train")
    output_path = Path("../../data/output/zero_padding_reduced/masks/sidu/train")
    output_path.mkdir(parents=True, exist_ok=True)

    # mask_generator = GradCamMaskGenerator(model.model)
    mask_generator = SiduMaskGenerator(model.model)
    generate_masks(images_path, output_path, model, mask_generator)
