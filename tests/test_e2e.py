import json
import tempfile
import time
from pathlib import Path

import pytest

from neptune_tensorflow_keras.impl import NeptuneCallback

try:
    # neptune-client=0.9.0+ package structure
    from neptune.new import init_run
except ImportError:
    # neptune-client>=1.0.0 package structure
    from neptune import init_run


def test_smoke(dataset, model):
    run = init_run()


    callback = NeptuneCallback(run=run)

    (x_train, y_train), (x_test, y_test) = dataset

    model.fit(
        x_train,
        y_train,
        epochs=5,
        callbacks=[callback],
        validation_data=(x_test, y_test),
    )

    base_namespace = "training"

    for subset in ["train", "test"]:
        for granularity in ["batch", "epoch"]:
            assert run.exists(f"{base_namespace}/{subset}/{granularity}")
            assert run.exists(f"{base_namespace}/{subset}/{granularity}/accuracy")
            assert run.exists(f"{base_namespace}/{subset}/{granularity}/loss")

    assert run.exists(f"{base_namespace}/model/summary")
