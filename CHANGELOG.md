## neptune-tensorflow-keras 2.0.0

### Changes
- `NeptuneCallback` now accepts namespace `Handler` as an alternative to `Run` for the `run` argument. This means that
  you can call it like `NeptuneCallback(run=run["some/namespace/"])` to log everything to the `some/namespace/`
  location of the run.

### Breaking changes
- Instead of the `log()` method, the integration now uses `append()` which is available since version 0.16.14
  of the neptune-client.

## neptune-tensorflow-keras 1.2.1

### Fixes
- Remove misleading information from the docstring. Pydot is needed when saving the model diagram.

## neptune-tensorflow-keras 1.2.0

### Changes
- Moved neptune_tensorflow_keras package to src dir ([#33](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/33))
- Poetry as a package builder ([#40](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/40))

### Fixes
- Fixed NeptuneCallback import error - now possible to directly import with `from neptune_tensorflow_keras import NeptuneCallback` ([#35](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/35))
- Drop usage of deprecated File.extension init attribute #38 ([#38](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/38))

## neptune-tensorflow-keras 1.1.0

### Fixes

- We fixed a bug that crashed the integration with an error message ``ValueError: This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.``, when the input layer of a neural network didn't have the `input_shape`
  parameter defined or the model was not built.

## neptune-tensorflow-keras 1.0.0

### Added

- We are additionally saving the model summary ([#14](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/14))
- We are saving the parameters of the optimizer used for training the model ([#15](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/15))
- We are saving the parameters passed to `Model.fit` during the training ([#17](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/17))
- We are logging the current learning rate at every epoch ([#18](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/18))
- You can use the `log_model_diagram=True` flag to save the model visualization produced by the Keras functions
  `model_to_dot` and `plot_model` ([#16](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/16))

### Changes

- Changed the utils of the integration to be imported from a non-internal package ([#24](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/24))
- To keep the metadata better organized, `NeptuneCallback` now logs everything according to the `base_namespace` argument,
  which defaults to `training`. `None` is no longer accepted as a value of the argument ([#14](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/14))
- Logging the batch metrics is now optional with the `log_on_batch` flag that defaults to False ([#19](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/19))
- To be consistent with the naming convention used by Keras, we changed the names of the logging directories for
  the validation metrics from "test" to "validation" ([#26](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/26))

### Fixes

- We fixed the dependencies for TensorFlow to version >= 2.0.0. ([#14](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/14))

## neptune-tensorflow-keras 0.9.9

### Features

- Added logging metrics in test and validation ([#9](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/9))

### Fixes

- Fixed when base_namespace set to None ([#8](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/8))
- Small fixes and updates to NeptuneCallback docstrings ([#6](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/6))

## neptune-tensorflow-keras 0.9.8

### Features

- Mechanism to prevent using legacy Experiments in new-API integrations ([#5](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/5))
