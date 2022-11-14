import numpy as np

from src.features.uiblungs.box.box import ImageBox
from src.features.uiblungs.cropper.cropper import BoxCropper


class BoxCropperImp(BoxCropper):

    def crop(self, image: np.ndarray, box: ImageBox) -> np.ndarray:
        return image[box.upper:box.lower, box.leftmost: box.rightmost]
