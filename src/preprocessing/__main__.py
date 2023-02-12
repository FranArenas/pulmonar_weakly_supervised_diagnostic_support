import random
from math import ceil
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tqdm

from preprocessing.entity.resize_mode import ResizeMode
from preprocessing.parser.arguments import Arguments
from preprocessing.parser.parser import Parser
from preprocessing.resizing.square.standard import StandardResizer
from preprocessing.resizing.square.zero_padding import ZeroPaddingResizer
from preprocessing.resource.reader.reader import ImageReader
from preprocessing.resource.reader.reader_impl import ImageReaderImpl
from preprocessing.resource.writer.writer import ImageWriter
from preprocessing.resource.writer.writer_impl import ImageWriterImpl


class Main:
    def __init__(self, arguments: Arguments, reader: ImageReader, writer: ImageWriter):
        self.arguments = arguments
        self.reader = reader
        self.writer = writer

        match self.arguments.resize_mode:
            case ResizeMode.STANDARD:
                self.resizer = StandardResizer()
            case ResizeMode.ZERO_PADDING:
                self.resizer = ZeroPaddingResizer()
            case _ as mode:
                raise ValueError(f"Mode {mode} not recognized")

    def run(self):
        if self.arguments.input_path.is_dir():

            train, test = self._train_test_split([*self.arguments.input_path.rglob("*.png")],
                                                 self.arguments.train_test_split)
            for file in tqdm.tqdm(train, desc="Train images"):
                image = self._read(file)
                if test:
                    self.writer.write(image,
                                      self.arguments.output_path.parent / "train" / self.arguments.output_path.name / file.name)
                else:
                    self.writer.write(image, self.arguments.output_path / file.name)

            for file in tqdm.tqdm(test, desc="Test images"):
                image = self._read(file)
                self.writer.write(image,
                                  self.arguments.output_path.parent / "test" / self.arguments.output_path.name / file.name)
        else:
            image = self._read(self.arguments.input_path)
            self.writer.write(image, self.arguments.output_path)

    def _read(self, path: str | Path) -> np.ndarray:
        return self.resizer.resize(self.reader.read(path, mode=self.arguments.color_mode),
                                   shape=self.arguments.shape)

    @staticmethod
    def _train_test_split(files: List[Path], percentage: float) -> Tuple[list, list]:
        if percentage == 0:
            return files, []
        train_split = random.sample(files, k=ceil(len(files) / (1 / percentage)))
        test_split = list(set(files) - set(train_split))
        return train_split, test_split


if __name__ == "__main__":
    arguments_impl = Parser().parse()
    reader_impl = ImageReaderImpl()
    writer_impl = ImageWriterImpl()

    Main(arguments_impl, reader_impl, writer_impl).run()
