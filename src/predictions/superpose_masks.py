from pathlib import Path

import cv2
from tqdm import tqdm

if __name__ == "__main__":
    masks_path = Path("../../data/unet/output")
    images_path = Path("../../data/unet/test/lung")
    outputs_path = Path("../../data/unet/superposed-output")

    masks_path = Path("../../data/unet/output")
    images_path = Path("../../data/unet/test/lung")
    outputs_path = Path("../../data/unet/superposed-output")

    images_paths = [*images_path.rglob("*.png")]

    for mask_path in tqdm([*masks_path.rglob("*.png")]):
        image_name = mask_path.name.split("-")[-1]
        image_path = [*filter(lambda x: image_name in str(x), images_paths)]
        assert len(image_path) == 1, f"expected: 1 | actual: {len(image_path)}"
        images_paths.remove(image_path[0])
        image_path = image_path[0].resolve()

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path.resolve()), cv2.IMREAD_GRAYSCALE)
        mask_coloured = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_coloured = cv2.GaussianBlur(mask_coloured, (13, 13), 11)
        super_imposed_img = cv2.addWeighted(mask_coloured, 0.5, image, 0.5, 0)

        output_path = (outputs_path / "/".join(mask_path.parts[-3:])).resolve()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if not cv2.imwrite(str(output_path), super_imposed_img):
            raise RuntimeError("The image was not written")
