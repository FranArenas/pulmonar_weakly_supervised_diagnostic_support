from pathlib import Path
from typing import Dict

import PIL.Image
import torch

from classification.resnet50 import Resnet50
from classification.transformation import basic_transformation


def predict(image_path: Path,
            model: Resnet50,
            device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, float]:
    image = torch.unsqueeze(basic_transformation(PIL.Image.open(image_path)), dim=0)

    if device == "cuda":
        image = image.cuda()

    result = model.predict(image).data.cpu().numpy()[0]
    predict_dict = {"covid": result[0], "nocovid": result[1]}
    return predict_dict


if __name__ == "__main__":
    image_path = Path(
        "/home/fran/myFiles/uib/cuarto/TFG/tfg_repo/data/output/zero_padding/test/nocovid/CRIS12992490.png")
    weights_path = Path(
        "/home/fran/myFiles/uib/cuarto/TFG/tfg_repo/data/weights/faa_tfg_resnet_weights_desc=Resnet50 standard 2048-1024-512-2__class 'classification.resnet50.Resnet50'_learning_rate=7.5e-05_weight_decay=0.00015_start_param_training=0_a53f9ea7-c004-42df-9a6a-75bb148c2af3.pt")

    model = Resnet50(torch.device("cpu"))
    model.load_weights(weights_path)

    print(predict(image_path, model))
