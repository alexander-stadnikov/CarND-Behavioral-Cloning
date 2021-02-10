from dataclasses import dataclass
from typing import List

@dataclass
class Frame:
    """
    The class represents a descriptor of a frame.

    An instance of the class contains all necessary data about a frame,
    such as paths to images, steering, throttle, brake and speed data.
    """

    center_img_path: str
    left_img_path: str
    right_img_path: str
    steering: float
    throttle: float
    brake: float
    speed: float

    def __init__(self, csv_line: List[str]):
        """ Initialize an instance from a record of a CSV file. """
        self.center_img_path = Frame._read_path(csv_line[0])
        self.left_img_path = Frame._read_path(csv_line[1])
        self.right_img_path = Frame._read_path(csv_line[2])
        self.steering = float(csv_line[3])
        self.throttle = float(csv_line[4])
        self.brake = float(csv_line[5])
        self.speed = float(csv_line[6])

    @staticmethod
    def _read_path(line: str) -> str:
        return line.replace('\\', '/')
