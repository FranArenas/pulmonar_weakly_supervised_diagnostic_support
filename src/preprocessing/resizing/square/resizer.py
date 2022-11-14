import abc

import numpy as np


class SquareImageResizer(abc.ABC):

    def resize(self,
               image: np.ndarray,
               shape: int) -> np.ndarray:
        ...
