## neptune-tensorflow-keras 1.0.0

### Changes

- `NeptuneCallback` now logs everything according to the `base_namespace` argument that defaults to `training`
  to keep the metadata better organized. `None` is not accepted anymore as a value of the argument.
- We are additionally saving the model summary.
- We fixed the dependencies for TensorFlow to version >= 2.0.0.

## neptune-tensorflow-keras 0.9.9

### Features
- Added logging metrics in test and validation ([#9](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/9))

### Fixes
- Fixed when base_namespace set to None ([#8](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/8))
- Small fixes and updates to NeptuneCallback docstrings ([#6](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/6))

## neptune-tensorflow-keras 0.9.8

### Features
- Mechanism to prevent using legacy Experiments in new-API integrations ([#5](https://github.com/neptune-ai/neptune-tensorflow-keras/pull/5))
