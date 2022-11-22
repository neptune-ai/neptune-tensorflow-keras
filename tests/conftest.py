import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def dataset():
    x_train = tf.random.uniform(shape=[2, 28, 28])
    y_train = tf.constant([1, 1], shape=(2, 1), dtype=tf.int8)
    x_test = tf.random.uniform(shape=[2, 28, 28])
    y_test = tf.constant([1, 1], shape=(2, 1), dtype=tf.int8)

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
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    return model
