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
    callback = NeptuneCallback(run=run, base_namespace="metrics")

    (x_train, y_train), (x_test, y_test) = dataset

    model.fit(x_train, y_train, epochs=5, callbacks=[callback])

    assert run.exists("metrics/train/batch")
    assert run.exists("metrics/train/batch/accuracy")
    assert run.exists("metrics/train/batch/loss")

    assert run.exists("metrics/train/epoch")
    assert run.exists("metrics/train/epoch/accuracy")
    assert run.exists("metrics/train/epoch/loss")

    assert run.exists("metrics/training/model/summary")
