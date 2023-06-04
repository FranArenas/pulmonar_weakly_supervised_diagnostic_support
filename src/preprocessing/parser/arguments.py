import pathlib

from pydantic import BaseModel, validator

from preprocessing.entity.color_mode import ColorMode
from preprocessing.entity.resize_mode import ResizeMode


class Arguments(BaseModel):
    shape: int
    input_path: pathlib.Path
    output_path: pathlib.Path
    resize_mode: ResizeMode
    train_test_split: float
    color_mode: ColorMode

    @validator("shape")
    def validate_shape(cls, v: int):
        if v <= 0:
            raise ValueError(f"The shape isn't > 0. Shape: {v}")
        return v

    @validator("input_path")
    def validate_input_path(cls, v: pathlib.Path):
        if not v.exists():
            raise FileNotFoundError(None, f"Input_path value ({v}) doesn't exists")
        return v

    @validator("train_test_split")
    def validate_train_test_split(cls, v: float):
        if v > 1.0 or v < 0.0:
            raise ValueError("Train mask split value should be between 1 and 0")
        return v
