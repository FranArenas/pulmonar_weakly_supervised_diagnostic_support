import numpy as np

from features.uiblungs.box.box import ImageBox
from features.uiblungs.cropper.cropper import BoxCropper


class BoxCropperImp(BoxCropper):

    def crop(self, image: np.ndarray, box: ImageBox) -> np.ndarray:
        return image[box.upper:box.lower, box.leftmost: box.rightmost]
