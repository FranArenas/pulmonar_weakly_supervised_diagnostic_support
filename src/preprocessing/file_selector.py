import random
import shutil
from pathlib import Path
from sys import argv

from tqdm import tqdm


def main(input_dir: Path, output_dir: Path, n: int):
    for file in tqdm(random.sample([*input_dir.glob("*")], n)):
        shutil.copy(file, output_dir / file.name)


if __name__ == "__main__":
    input_dir = Path(argv[1]).resolve()
    output_dir = Path(argv[2]).resolve()
    n = int(argv[3])

    if not input_dir.is_dir():
        raise ValueError(f"The value {input_dir} is not a directory")

    if n < 0:
        raise ValueError(f"Value {n} should be greater than 0")

    if n > (val := len([*input_dir.glob("*")])):
        raise ValueError(f"Value {n} is bigger than the number of"
                         f" files in the directory {val}")

    output_dir.mkdir(parents=True, exist_ok=True)

    main(input_dir, output_dir, n)
