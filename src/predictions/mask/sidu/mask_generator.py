import PIL.Image
import cv2
import numpy as np
import torch

from predictions.mask.mask_generator import MaskGenerator
from predictions.mask.sidu.sidu_memory import sidu_wrapper
from predictions.models.transformation import basic_transformation


class SiduMaskGenerator(MaskGenerator):
    def __init__(self, model, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

    def generate_mask(self, image: PIL.Image) -> np.ndarray:
        tensor = torch.unsqueeze(basic_transformation(image), dim=0)
        tensor = tensor.cuda() if self.device == "cuda" else tensor.cpu()
        sidu_output: np.ndarray = sidu_wrapper(self.model, self.model.layer4[2].conv3, tensor)
        transposed_heatmap = np.transpose(sidu_output.squeeze(0), (1, 2, 0))
        normalized_heatmap = np.zeros((image.size[0], image.size[1]))
        normalized_heatmap = cv2.normalize(transposed_heatmap, normalized_heatmap, 0, 1, cv2.NORM_MINMAX)

        return normalized_heatmap
