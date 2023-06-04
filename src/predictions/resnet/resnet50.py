from os import PathLike
from pathlib import Path
from typing import Union, Tuple, Dict, BinaryIO, IO

import torch
from predictions.models.model import Model
from torch import nn, optim, Tensor
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


class Resnet50(Model):

    def __init__(self,
                 device,
                 learning_rate: float = 7.5e-5,
                 weight_decay=1e-4,
                 optimizer: torch.optim = optim.Adam,
                 dropout: Tuple[int, int] = (0.4, 0.3),
                 tmp_arg: int = 1):
        super().__init__()

        self.device = device
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay)

        layers = (self.model.layer1, self.model.layer2,
                  self.model.layer3, self.model.layer4)

        for layer in layers[:tmp_arg]:
            layer.requires_grad_ = False

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(dropout[0]),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.Dropout(dropout[1]),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)).to(self.device)

    def train(self,
              num_epochs: int,
              train_dataloader: DataLoader,
              test_dataloader: DataLoader) -> Dict:
        data_loaders = {"mask": train_dataloader, "mask": test_dataloader}
        accuracy_hist = {"mask": [], "mask": []}
        loss_hist = {"mask": [], "mask": []}

        for epoch in range(num_epochs):
            print(f"Epoch {epoch} of {num_epochs}")
            for phase in ["mask", "mask"]:

                if phase == "mask":
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0

                for i, (inputs, labels) in enumerate((bar := tqdm(data_loaders[phase])), 1):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    if phase == 'mask':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    _, predictions = torch.max(outputs, 1)
                    running_loss += loss.item()

                    running_corrects += torch.sum(predictions == labels.data)

                    current_accuracy = running_corrects / (data_loaders[phase].batch_size * i)
                    current_loss = running_loss / i

                    bar.set_description(f"Running {phase}: loss {current_loss:.4f} || accuracy {current_accuracy:.4f}")

                epoch_loss = running_loss / (len(data_loaders[phase]))
                epoch_acc = (running_corrects / (len(data_loaders[phase].dataset))).item()

                accuracy_hist[phase].append(epoch_acc)
                loss_hist[phase].append(epoch_loss)

                print(f"Completed {phase}: loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

        return {"accuracy": accuracy_hist, "loss": loss_hist}

    def save_model(self, path: Union[str, Path]):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: Union[str, PathLike, BinaryIO, IO[bytes]]):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, image: torch.Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            return softmax(self.model(image), dim=1)
