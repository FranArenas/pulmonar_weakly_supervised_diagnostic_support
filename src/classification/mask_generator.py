import PIL.Image
import cv2
import numpy as np
import torch
from captum.attr import LayerGradCam, LayerAttribution

from classification.transformation import basic_transformation


class GradCamMaskGenerator:
    def __init__(self, model):
        self.model = model

    def generate_mask(self, image: PIL.Image) -> np.ndarray:
        tensor = torch.unsqueeze(basic_transformation(image), dim=0)
        grad_cam = LayerGradCam(self.model, self.model.layer4[2].conv3)
        attr = grad_cam.attribute(tensor, 1, relu_attributions=True)
        upsampled_attr = LayerAttribution.interpolate(attr, (800, 800))
        output_heatmap = upsampled_attr.detach().numpy()[0]
        transposed_heatmap = np.transpose(output_heatmap, (1, 2, 0))
        normalized_heatmap = np.zeros((image.size[0], image.size[1]))
        normalized_heatmap = cv2.normalize(transposed_heatmap, normalized_heatmap, 0, 1, cv2.NORM_MINMAX)

        return normalized_heatmap
