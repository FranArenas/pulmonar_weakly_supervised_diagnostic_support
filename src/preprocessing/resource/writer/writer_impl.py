from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.resource.writer.writer import ImageWriter


class ImageWriterImpl(ImageWriter):
    def write(self, image: np.ndarray, filepath: Path | str):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), image)
