import pytest

from neptune_tensorflow_keras.impl import NeptuneCallback

try:
    # neptune-client=0.9.0+ package structure
    from neptune.new import init_run
except ImportError:
    # neptune-client>=1.0.0 package structure
    from neptune import init_run


@pytest.mark.parametrize("log_model_diagram", [True, False])
def test_smoke(dataset, model, log_model_diagram):
    run = init_run()

    callback = NeptuneCallback(run=run, log_model_diagram=log_model_diagram)

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

    if log_model_diagram:
        assert run.exists(f"{base_namespace}/model/visualization")
        assert run[f"{base_namespace}/model/visualization"].fetch_extension() == "png"
    else:
        assert not run.exists(f"{base_namespace}/model/visualization")
