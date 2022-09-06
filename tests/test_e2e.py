import numpy.testing as npt

from neptune_tensorflow_keras.impl import NeptuneCallback

try:
    # neptune-client=0.9.0+ package structure
    from neptune.new import init_run
except ImportError:
    # neptune-client>=1.0.0 package structure
    from neptune import init_run


@pytest.mark.parametrize("log_on_batch", [True, False])
def test_smoke(dataset, model, log_on_batch):
    run = init_run()

    callback = NeptuneCallback(run=run, log_on_batch=log_on_batch)

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
            if granularity == "batch" and not log_on_batch:
                assert not run.exists(f"{base_namespace}/{subset}/{granularity}")
            else:
                assert run.exists(f"{base_namespace}/{subset}/{granularity}")
                assert run.exists(f"{base_namespace}/{subset}/{granularity}/accuracy")
                assert run.exists(f"{base_namespace}/{subset}/{granularity}/loss")

    assert run.exists(f"{base_namespace}/model/summary")

    assert run.exists(f"{base_namespace}/model/optimizer_config")
    assert run[f"{base_namespace}/model/optimizer_config/name"].fetch() == "SGD"
    npt.assert_approx_equal(run[f"{base_namespace}/model/optimizer_config/learning_rate"].fetch(), 0.01)
    npt.assert_approx_equal(run[f"{base_namespace}/model/optimizer_config/decay"].fetch(), 0)
    npt.assert_approx_equal(run[f"{base_namespace}/model/optimizer_config/momentum"].fetch(), 0)
    assert run[f"{base_namespace}/model/optimizer_config/nesterov"].fetch() == False
