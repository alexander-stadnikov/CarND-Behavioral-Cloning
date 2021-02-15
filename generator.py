from typing import List
import numpy as np
from sklearn.utils import shuffle

from frame import Frame


def generator(samples: List[Frame], batch_size: int) -> List[np.ndarray]:
    """
    Functional generator of training data.
    Splits whole training set onto batches and returns augmented data.

    Args:
        samples: Test samples.
        batch_size: The size of a batch.
    """
    while True:
        shuffle(samples)
        for start in range(0, len(samples), batch_size):
            end = start + batch_size
            batch_samples = samples[start:end]
            images = []
            angles = []

            for frame in batch_samples:
                for i, s in [frame.center(), frame.right(), frame.left()]:
                    images.append(i)
                    angles.append(s)

            yield np.array(images), np.array(angles)
