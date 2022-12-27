import argparse
from typing import Sequence, Optional

from src.preprocessing.entity.resize_mode import ResizeMode
from src.preprocessing.parser.arguments import Arguments


class Parser:

    @staticmethod
    def parse(arguments: Optional[Sequence[str]] = None) -> Arguments:
        parser = argparse.ArgumentParser()

        parser.add_argument("-shape", type=int, required=True)
        parser.add_argument("-input_path", type=str, required=True)
        parser.add_argument("-output_path", type=str, required=True)
        parser.add_argument("--resize_mode", type=str,
                            choices=[val.value for val in ResizeMode], default="STANDARD")
        parser.add_argument("--train_test_split", type=float, default=0.0)

        args = parser.parse_args(arguments)

        return Arguments(shape=args.shape,
                         input_path=args.input_path,
                         output_path=args.output_path,
                         resize_mode=args.resize_mode,
                         train_test_split=args.train_test_split)
