import tqdm

from preprocessing.entity.resize_mode import ResizeMode
from preprocessing.parser.parser import Parser
from preprocessing.resizing.square.standard import StandardResizer
from preprocessing.resizing.square.zero_padding import ZeroPaddingResizer
from preprocessing.resource.reader.reader_impl import ImageReaderImpl
from preprocessing.resource.writer.writer_impl import ImageWriterImpl

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
        for file in tqdm.tqdm([*arguments.input_path.iterdir()]):
            image = resizer.resize(reader.read(file), shape=arguments.shape)
            writer.write(image, arguments.output_path / file.name)
    else:
        image = resizer.resize(reader.read(arguments.input_path), shape=arguments.shape)
        writer.write(image, arguments.output_path)
