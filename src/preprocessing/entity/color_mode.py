from enum import Enum

import cv2


class ColorMode(Enum):
    GRAY_SCALE = cv2.IMREAD_GRAYSCALE
    COLOR = cv2.IMREAD_COLOR
