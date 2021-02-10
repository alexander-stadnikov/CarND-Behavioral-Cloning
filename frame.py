from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class Frame:
    """
    The class represents a descriptor of a frame.

    An instance of the class contains all necessary data about a frame,
    such as paths to images, steering, throttle, brake and speed data.
    """

    center_img_path: Path
    left_img_path: Path
    right_img_path: Path
    steering: float
    throttle: float
    brake: float
    speed: float

    def __init__(self, csv_file_path: str, csv_line: List[str]):
        """ Initialize an instance from a record of a CSV file. """
        self.center_img_path = Frame._read_path(csv_file_path, csv_line[0])
        self.left_img_path = Frame._read_path(csv_file_path, csv_line[1])
        self.right_img_path = Frame._read_path(csv_file_path, csv_line[2])
        self.steering = float(csv_line[3])
        self.throttle = float(csv_line[4])
        self.brake = float(csv_line[5])
        self.speed = float(csv_line[6])

    @staticmethod
    def _read_path(file: Path, img_path: str) -> str:
        return (Path(file).parent / img_path).resolve()
