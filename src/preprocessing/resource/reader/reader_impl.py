from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.resource.reader.reader import ImageReader


class ImageReaderImpl(ImageReader):
    def read(self, filepath: str | Path) -> np.ndarray:
        if Path(filepath).is_dir():
            raise IsADirectoryError(f"{filepath} is a directory")

        return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
