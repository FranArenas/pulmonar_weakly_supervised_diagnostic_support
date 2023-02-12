import numpy as np
from skimage.transform import resize

from preprocessing.resizing.square.resizer import SquareImageResizer


class StandardResizer(SquareImageResizer):
    def resize(self,
               image: np.ndarray,
               shape: int,
               anti_aliasing: bool = True) -> np.ndarray:
        return resize(image,
                      output_shape=(shape, shape),
                      preserve_range=True,
                      anti_aliasing=anti_aliasing)
