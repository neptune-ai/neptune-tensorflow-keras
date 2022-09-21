## [UNRELEASED] neptune-tensorflow-keras 1.0.0

### Added

- We are additionally saving the model summary ([#14](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/14))
- We are saving the parameters of the optimizer used for training the model ([#15](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/15))
- We are saving the parameters passed to `Model.fit` during the training ([#17](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/17))
- We are logging the current learning rate at every epoch ([#18](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/18))
- You can use the `log_model_diagram=True` flag to save the model visualization produced by `model_to_dot` and 
  `plot_model` by Keras ([#16](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/16))

### Changes

- Changed integrations utils to be imported from non-internal package ([#24](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/24))
- `NeptuneCallback` now logs everything according to the `base_namespace` argument that defaults to `training`
  to keep the metadata better organized. `None` is not accepted anymore as a value of the argument ([#14](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/14))
- Logging the batch metrics is now optional with the `log_on_batch` flag that defaults to False ([#19](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/19))
- We changed the names of the validation metrics logging directories from "test" to "validation" to be consistent with
  the naming convention used by Keras ([#26](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/26))

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
