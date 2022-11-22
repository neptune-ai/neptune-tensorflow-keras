import random

import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_point = random.randint(1, len(x_train))
    test_point = random.randint(1, len(x_test))

    x_train = x_train[train_point - 1 : train_point + 1, :, :]
    y_train = y_train[train_point - 1 : train_point + 1]

    x_test = x_test[test_point - 1 : test_point + 1, :, :]
    y_test = y_test[test_point - 1 : test_point + 1]

    return (x_train, y_train), (x_test, y_test)


@pytest.fixture(scope="session")
def model():
    model = tf.keras.models.Sequential(
        [
            # We are *not* providing input_size to the first layer, so the test will catch also the case where the
            # model was not build yet: https://stackoverflow.com/q/55908188/3986320
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="sgd", loss=loss_fn, metrics=["accuracy"])
    return model
