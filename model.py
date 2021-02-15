from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D, Activation
from keras.layers import Lambda, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np

from typing import List, Tuple
import os

from data_import import read_csv
from frame import Frame


import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Read collected input data
samples = []
samples_sources = ['my_direct', 'my_reverse']
# samples_sources = ['udacity']
for src in samples_sources:
    samples.extend(read_csv(f"./data/{src}/driving_log.csv", speed_limit=0.1))

# Split all sample onto validation and test samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples: List[Frame], batch_size: int) -> List[np.ndarray]:
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

            yield np.array(images), np.array(angles) #here we do not hold the values of X_train and y_train instead we yield the values which means we hold until the generator is running


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# Create the model
def Dave2CNN() -> Sequential:
    """
    Creates the DAVE-2 CNN. The network invented by nVidia.

    For more informaion, plase, read this blog post:
    https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    """
    model = Sequential(
        layers= [
            Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3), name='Normalization'),
            Cropping2D(cropping=((60, 25), (0, 0))),
            Conv2D(24, 5, strides=2, activation='elu', padding='same'),
            Conv2D(36, 5, strides=2, activation='elu', padding='same'),
            Conv2D(48, 5, strides=2, activation='elu', padding='same'),
            Conv2D(64, 3, activation='elu', padding='same'),
            Conv2D(64, 3, activation='elu', padding='same'),
            Flatten(),
            Dense(100, activation='elu'),
            Dense(50, activation='elu'),
            Dense(10, activation='elu'),
            Dense(1)
        ],
        name="DAVE2"
    )

    model.compile(optimizer='adam', loss='mse')
    return model

model = Dave2CNN()
model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=0
    ),
    ModelCheckpoint(
        filepath=os.path.join('output/model_sim_{epoch:02d}.hdf5'),
        monitor='val_loss',
        verbose=0,
        save_best_only=False
    )
]

model.fit(
    train_generator,
    steps_per_epoch=len(train_samples) // 64,
    validation_data=validation_generator,
    validation_steps=len(validation_samples) // 64,
    epochs=5,
    # callbacks=callbacks,
    verbose=1
)

model.save('output/model.h5')
