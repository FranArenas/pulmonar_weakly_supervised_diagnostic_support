import pathlib

import torch
from predictions.models.transformation import train_transformation, basic_transformation
from predictions.resnet50 import Resnet50
from torch import device
from torchvision import datasets


def train():
    input_path = pathlib.Path("../../data/preprocessed/zero_padding").resolve()
    weights_path = pathlib.Path("../../data/weights/zero_padding/weights.h5").resolve()

    train_dataset = datasets.ImageFolder(str(input_path / 'mask'), train_transformation)
    test_dataset = datasets.ImageFolder(str(input_path / 'mask'), basic_transformation)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=3,
                                                   shuffle=True,
                                                   num_workers=0)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=3,
                                                  shuffle=True,
                                                  num_workers=0)

    model = Resnet50(device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Train started")
    model.train(num_epochs=2, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    model.save_model(weights_path)


if __name__ == "__main__":
    train()
