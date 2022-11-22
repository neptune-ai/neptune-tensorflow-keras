import numpy as np
import pytest
import tensorflow as tf


def _vectorize_sequences(sequences, dims=1000):
    results = np.zeros((len(sequences), dims))
    for idx, seq in enumerate(sequences):
        results[idx, seq] = 1.0

    return results


@pytest.fixture(scope="session")
def dataset():
    imdb = tf.keras.datasets.imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

    x_train = _vectorize_sequences(x_train)
    x_test = _vectorize_sequences(x_test)

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
