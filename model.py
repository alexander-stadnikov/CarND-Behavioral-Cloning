from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import Lambda, Cropping2D

from data_import import read_csv


# Read collected input data
samples = []
samples_sources = ['my_direct', 'my_reverse', 'udacity']
for src in samples_sources:
    samples.extend(read_csv(f"./data/{src}/driving_log.csv", speed_limit=0.1))

# Create the model
def Dave2CNN() -> Sequential:
    """
    Creates the DAVE-2 CNN. The network invented by nVidia.

    For more informaion, plase, read this blog post:
    https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    """
    def _convolution(filter_size: int, kernel_size: int) -> Conv2D:
        return Conv2D(
            filter_size,
            kernel_size,
            strides=(2, 2),
            activation='elu',
            padding='same'
        )

    model = Sequential(
        [
            Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3), name='Normalization'),
            Cropping2D(cropping=((60, 25), (0, 0))),
            _convolution(24, 5),
            _convolution(36, 5),
            _convolution(48, 5),
            _convolution(64, 3),
            _convolution(64, 3),
            Flatten(),
            Dense(100, activation='elu'),
            Dense(50, activation='elu'),
            Dense(10, activation='elu'),
            Dense(1)
        ]
    )

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    return model

model = Dave2CNN()
