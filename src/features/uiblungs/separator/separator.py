import abc

import numpy as np

from features.uiblungs.box.box import ImageBox


class LungSeparator(abc.ABC):

    @abc.abstractmethod
    def get_boxes(self, image: np.ndarray) -> dict[str, ImageBox]:
        ...
