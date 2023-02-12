import argparse
from typing import Sequence, Optional

from preprocessing.entity.color_mode import ColorMode
from preprocessing.entity.resize_mode import ResizeMode
from preprocessing.parser.arguments import Arguments


class Parser:

    @staticmethod
    def parse(arguments: Optional[Sequence[str]] = None) -> Arguments:
        parser = argparse.ArgumentParser()

        parser.add_argument("--shape", type=int, required=True)
        parser.add_argument("--input_path", type=str, required=True)
        parser.add_argument("--output_path", type=str, required=True)
        parser.add_argument("--resize_mode", type=str,
                            choices=[val.value for val in ResizeMode], default="STANDARD")
        parser.add_argument("--train_test_split", type=float, default=0.0)
        parser.add_argument("--color_mode", type=str, choices=[val.value for val in ColorMode],
                            default=ColorMode.COLOR)

        args = parser.parse_args(arguments)

        return Arguments(shape=args.shape,
                         input_path=args.input_path,
                         output_path=args.output_path,
                         resize_mode=args.resize_mode,
                         train_test_split=args.train_test_split,
                         color_mode=args.color_mode)
