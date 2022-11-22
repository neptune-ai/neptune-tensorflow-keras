import time

import numpy.testing as npt
import pytest

from neptune_tensorflow_keras.impl import NeptuneCallback

try:
    # neptune-client=0.9.0+ package structure
    from neptune.new import init_run
    from neptune.new.exceptions import FetchAttributeNotFoundException
except ImportError:
    # neptune-client>=1.0.0 package structure
    from neptune import init_run
    from neptune.exceptions import FetchAttributeNotFoundException


@pytest.mark.parametrize("log_model_diagram", [True, False])
@pytest.mark.parametrize("log_on_batch", [True, False])
def test_e2e(dataset, model, log_model_diagram, log_on_batch):
    run = init_run()

    callback = NeptuneCallback(run=run, log_model_diagram=log_model_diagram, log_on_batch=log_on_batch)

    (x_train, y_train), (x_test, y_test) = dataset

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=1,
        callbacks=[callback],
        validation_data=(x_test, y_test),
    )

    # retry if Neptune didn't refresh its cache
    num_tries = 5
    for i in range(num_tries):
        try:
            validate_results(run, log_model_diagram, log_on_batch)
            return
        except FetchAttributeNotFoundException:
            time.sleep(i + 1)
    else:
        raise RuntimeError("Test failed to fetch the data from Neptune")


def validate_results(run, log_model_diagram, log_on_batch):
    base_namespace = "training"

    for subset in ["train", "validation"]:
        for granularity in ["batch", "epoch"]:
            if granularity == "batch" and not log_on_batch:
                assert not run.exists(f"{base_namespace}/{subset}/{granularity}")
            else:
                assert run.exists(f"{base_namespace}/{subset}/{granularity}")
                assert run.exists(f"{base_namespace}/{subset}/{granularity}/accuracy")
                assert run.exists(f"{base_namespace}/{subset}/{granularity}/loss")

    assert run.exists(f"{base_namespace}/model/summary")
    assert run.exists(f"{base_namespace}/model/learning_rate")

    assert run.exists(f"{base_namespace}/model/optimizer_config")
    assert run[f"{base_namespace}/model/optimizer_config/name"].fetch() == "SGD"
    learning_rate_dict = run[f"{base_namespace}/model/optimizer_config/learning_rate"].fetch()
    assert learning_rate_dict["config"]["decay_rate"] == 0.9
    assert learning_rate_dict["config"]["initial_learning_rate"] == 0.01

    npt.assert_approx_equal(run[f"{base_namespace}/model/optimizer_config/momentum"].fetch(), 0)
    assert run[f"{base_namespace}/model/optimizer_config/nesterov"].fetch() is False

    assert run.exists(f"{base_namespace}/fit_params")
    assert run.exists(f"{base_namespace}/fit_params/epochs")
    assert run[f"{base_namespace}/fit_params/epochs"].fetch() == 5

    if log_model_diagram:
        assert run.exists(f"{base_namespace}/model/visualization")
        assert run[f"{base_namespace}/model/visualization"].fetch_extension() == "png"
    else:
        assert not run.exists(f"{base_namespace}/model/visualization")
