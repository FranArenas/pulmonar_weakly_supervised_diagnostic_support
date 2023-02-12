from pathlib import Path

import cv2
import numpy as np

from preprocessing.entity.color_mode import ColorMode
from preprocessing.resource.reader.reader import ImageReader


class ImageReaderImpl(ImageReader):
    def read(self, filepath: str | Path, mode: ColorMode = ColorMode.COLOR) -> np.ndarray:
        if Path(filepath).is_dir():
            raise IsADirectoryError(f"{filepath} is a directory")

        return cv2.imread(str(filepath), mode.value)
