from typing import List, Tuple
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Frame:
    """
    The class represents a descriptor of a frame.

    An instance of the class contains all necessary data about a frame,
    such as paths to images, steering, throttle, brake and speed data.
    """

    def __init__(self,
                 csv_file_path: str,
                 csv_line: List[str],
                 flipping: bool = False,
                 fading: bool = False,
                 steering_correction: float = 0.25):
        """
        Initialize an instance from a record of a CSV file.

        Args:
            csv_file_path: The path to the CSV file.
            csv_line: The line number from the CSV file.
            flipping: Whether or not flipping must be used.
            steering_correction: The steering correction for left and right cameras.
        """
        self.center_img_path = Frame._read_path(csv_file_path, csv_line[0])
        self.left_img_path = Frame._read_path(csv_file_path, csv_line[1])
        self.right_img_path = Frame._read_path(csv_file_path, csv_line[2])
        self.steering = float(csv_line[3])
        self.throttle = float(csv_line[4])
        self.brake = float(csv_line[5])
        self.speed = float(csv_line[6])
        self.flipping = flipping
        self.fading = fading
        self.steering_correction = steering_correction

    def center(self) -> Tuple[np.ndarray, float]:
        """
        Returns the center image and the corresponding steering angle.
        """
        return self._augment(Frame._img(self.center_img_path), self.steering)

    def left(self) -> Tuple[np.ndarray, float]:
        """
        Returns the left image and the corresponding steering angle.
        """
        return self._augment(Frame._img(self.left_img_path),
            self.steering + self.steering_correction)

    def right(self) -> Tuple[np.ndarray, float]:
        """
        Returns the right image and the corresponding steering angle.
        """
        return self._augment(Frame._img(self.right_img_path),
            self.steering - self.steering_correction)

    def _augment(
        self,
        img: np.ndarray,
        steering: float,
        debug: bool = False) -> Tuple[np.ndarray, float]:

        augmented = False

        if debug:
            img_orig = img.copy()

        if self.flipping:
            img, steering = self._flip(img, steering)
            augmented = True

        if self.fading:
            img, steering = self._fade(img, steering)
            augmented = True

        if debug and augmented:
            plt.axis('off')
            plt.subplot(1, 2, 1)
            plt.imshow(img_orig)
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.show()

        return img, steering

    def _flip(self, img: np.ndarray, steering: float) -> Tuple[np.ndarray, float]:
        return cv2.flip(img, 1), -steering

    def _fade(self, img: np.ndarray, steering: float) -> Tuple[np.ndarray, float]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = np.array(img, dtype=np.float64)
        bright = .5 + np.random.uniform()
        img[:,:,2] = img[:,:,2]*bright
        img[:,:,2][img[:,:,2]>255] = 255
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, steering

    @staticmethod
    def _read_path(file: Path, img_path: str) -> str:
        return (Path(file).parent / img_path).resolve()

    @staticmethod
    def _img(path: Path) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imread(str(path)),
            cv2.COLOR_BGR2RGB
        )
