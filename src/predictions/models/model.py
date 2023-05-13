import abc
from os import PathLike
from pathlib import Path
from typing import Dict, Union, BinaryIO, IO

from torch import Tensor
from torch.utils.data import DataLoader


class Model(abc.ABC):
    @abc.abstractmethod
    def train(self,
              num_epochs: int,
              train_dataloader: DataLoader,
              test_dataloader: DataLoader) -> Dict:
        ...

    def save_model(self, path: Union[str, Path]):
        ...

    def load_weights(self, path: Union[str, PathLike, BinaryIO, IO[bytes]]):
        ...

    def predict(self, image: Tensor) -> Tensor:
        ...
