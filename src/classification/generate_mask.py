from pathlib import Path

import PIL.Image
import cv2
from tqdm import tqdm

from classification.mask_generator import GradCamMaskGenerator
from classification.predict import predict
from classification.resnet50 import Resnet50


def generate_masks(images_path: Path,
                   masks_path: Path,
                   model: Resnet50):
    mask_generator = GradCamMaskGenerator(model.model)
    labels_paths = [*images_path.glob("*")]

    for label_path in labels_paths:
        output_path = masks_path / label_path.name
        output_path.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm([*label_path.glob("*")], desc=f"Generating {label_path.name} masks"):

            prediction = predict(image_path, model)

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
        "../../data/weights/desc=test_acc=0.8310810923576355_test_loss=0.48300980299711227_Resnet50 2048-1024-512_Resnet50learning_rate=7.5e-05_weight_decay=0.001_start_param_training=30_nepochs=35_eafa44ca-c07e-4304-a90f-bee1e521.pt")
    model = Resnet50("cpu")
    model.load_weights(weights)

    images_path = Path("../../data/output/zero_padding/train")
    output_path = Path("../../data/output/zero_padding/masks/train")
    output_path.mkdir(parents=True, exist_ok=True)

    generate_masks(images_path, output_path, model)
