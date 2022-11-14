import argparse
import sys

from src.features.uiblungs.cropper.cropper_imp import BoxCropperImp
from src.features.uiblungs.separator.separator_imp import LungSeparatorImp
from src.features.uiblungs.slicer.slicer_imp import BoxSlicerImp
from src.features.uiblungs.splitter import Splitter


class Main:

    def __init__(self, spliter: Splitter):
        self.splitter = spliter

        parser = argparse.ArgumentParser()
        parser.add_argument('image', help="Lungs image")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)


if __name__ == "__main__":
    Main(Splitter(
        separator=LungSeparatorImp(),
        cropper=BoxCropperImp(),
        slicer=BoxSlicerImp()
    ))
