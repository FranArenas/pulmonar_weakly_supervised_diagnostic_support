import random
from math import ceil
from pathlib import Path
from typing import Tuple, Iterable, List

import tqdm

from src.preprocessing.entity.resize_mode import ResizeMode
from src.preprocessing.parser.parser import Parser
from src.preprocessing.resizing.square.standard import StandardResizer
from src.preprocessing.resizing.square.zero_padding import ZeroPaddingResizer
from src.preprocessing.resource.reader.reader_impl import ImageReaderImpl
from src.preprocessing.resource.writer.writer_impl import ImageWriterImpl


def _train_test_split(files: List[Path], percentage: float) -> Tuple[Iterable, Iterable]:
    train_split = random.choices(files, k=ceil(len(files) / percentage))
    test_split = [x for x in files if x not in train_split]
    return train_split, test_split


if __name__ == "__main__":
    arguments = Parser().parse()
    reader = ImageReaderImpl()
    writer = ImageWriterImpl()

    match arguments.resize_mode:
        case ResizeMode.STANDARD:
            resizer = StandardResizer()
        case ResizeMode.ZERO_PADDING:
            resizer = ZeroPaddingResizer()
        case _ as mode:
            raise ValueError(f"Mode {mode} not recognized")

    if arguments.input_path.is_dir():

        train, test = _train_test_split(list(arguments.input_path.iterdir()),
                                        arguments.train_test_split)
        for file in tqdm.tqdm(train, desc="Train images"):
            image = resizer.resize(reader.read(file), shape=arguments.shape)
            writer.write(image, arguments.output_path / "train" / file.name)

        for file in tqdm.tqdm(test, desc="Test images"):
            image = resizer.resize(reader.read(file), shape=arguments.shape)
            writer.write(image, arguments.output_path / "test" / file.name)
    else:
        image = resizer.resize(reader.read(arguments.input_path), shape=arguments.shape)
        writer.write(image, arguments.output_path)
