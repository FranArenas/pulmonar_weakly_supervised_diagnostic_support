import abc

import PIL.Image
import numpy as np


class MaskGenerator(abc.ABC):
    def generate_mask(self, image: PIL.Image) -> np.ndarray:
        ...
