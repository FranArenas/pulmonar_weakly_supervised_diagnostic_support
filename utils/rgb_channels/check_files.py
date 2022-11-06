import multiprocessing.pool
from pathlib import Path

import cv2
import tqdm

from utils.rgb_channels.check_file import check_file


def main(directory: Path):
    files = [str(file.resolve()) for file in directory.iterdir()]
    with multiprocessing.Pool() as pool:
        with tqdm.tqdm(total=len(files)) as pbar:
            for _ in pool.imap_unordered(check_file, files):
                pbar.update()

    print("All channels have the same value")


if __name__ == "__main__":
    asd = cv2.imread(str(Path("../../data/input/deleted_covid/CRIS13644609.png").resolve()), cv2.IMREAD_GRAYSCALE)
    main(Path("../../data/input/nocovid").resolve())
