from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__":
    input_dir_path = Path("../../../data/hands-lungs/gradcam").resolve()
    output_dir_path = Path("../../../data/hands-lungs/gradcam-corrected").resolve()

    for image_path in input_dir_path.rglob("*png"):
        image = cv2.imread(str(image_path.resolve()), cv2.IMREAD_UNCHANGED)
        discrete_image = np.zeros_like(image)
        discrete_image[image > 100] = 255
        mask_new_path = Path(f"output_path/{image_path.parent.parent}/{image_path.name}")
        mask_new_path.parent.mkdir(parents=True, exist_ok=True)

        output_filepath = Path(f"{output_dir_path}/{image_path.parent.parent.name}/"
                               f"{image_path.name}").resolve()
        output_filepath.parent.mkdir(exist_ok=True, parents=True)

        if not cv2.imwrite(str(output_filepath), discrete_image):
            raise FileExistsError()
