import math

import cv2
import numpy as np
from skimage.transform import resize

from src.preprocessing.resizing.square.resizer import SquareImageResizer


class ZeroPaddingResizer(SquareImageResizer):
    def resize(self,
               image: np.ndarray,
               shape: int,
               anti_aliasing: bool = True) -> np.ndarray:
        image_without_padding = self._resize_without_padding(image, shape, anti_aliasing)

        return self._add_zero_padding(image_without_padding)

    @staticmethod
    def _resize_without_padding(image: np.ndarray,
                                shape: int,
                                anti_aliasing: bool) -> np.ndarray:

        if image.shape[0] > image.shape[1]:
            resized_shape = (shape,
                             round(shape * image.shape[1] / image.shape[0]))
        elif image.shape[0] < image.shape[1]:
            resized_shape = (round(shape * image.shape[0] / image.shape[1]),
                             shape)
        else:
            resized_shape = (shape, shape)

        return resize(image=image,
                      output_shape=resized_shape,
                      preserve_range=True,
                      anti_aliasing=anti_aliasing)

    @staticmethod
    def _add_zero_padding(image: np.ndarray) -> np.ndarray:

        difference_pixels = abs((image.shape[1] - image.shape[0]) / 2)

        if image.shape[0] < image.shape[1]:
            return cv2.copyMakeBorder(image,
                                      top=math.ceil(difference_pixels),
                                      bottom=math.floor(difference_pixels),
                                      left=0,
                                      right=0,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)

        return cv2.copyMakeBorder(image,
                                  top=0,
                                  bottom=0,
                                  left=math.ceil(difference_pixels),
                                  right=math.floor(difference_pixels),
                                  borderType=cv2.BORDER_CONSTANT,
                                  value=0)
