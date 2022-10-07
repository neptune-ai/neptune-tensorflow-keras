# Neptune + TensorFlow/Keras Integration

Experiment tracking, model registry, data versioning, and live model monitoring for TensorFlow/Keras trained models.

## What will you get with this integration?

* Log, display, organize, and compare ML experiments in a single place
* Version, store, manage, and query trained models, and model building metadata
* Record and monitor model training, evaluation, or production runs live
* Collaborate with a team

## What will be logged to Neptune?

* hyperparameters for every run,
* learning curves for losses and metrics during training,
* hardware consumption and stdout/stderr output during training,
* TensorFlow tensors as images to see model predictions live,
* training code and git commit information,
* model weights
* [other metadata](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)

![image](https://user-images.githubusercontent.com/97611089/160638338-8a276866-6ce8-4d0a-93f5-bd564d00afdf.png)
*Example charts in the Neptune UI with logged accuracy and loss*


## Resources

* [Documentation](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras)
* [Code example on GitHub](https://github.com/neptune-ai/examples/blob/main/integrations-and-supported-tools/tensorflow-keras/scripts)
* [Runs logged in the Neptune app](https://app.neptune.ai/o/common/org/tf-keras-integration/e/TFK-18/all)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/tensorflow-keras/notebooks/Neptune_TensorFlow_Keras.ipynb)

## Example

```python
# On the command line:
pip install tensorflow neptune-client neptune-tensorflow-keras
```
```python
# In Python:
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


# Start a run
run = neptune.init(project="common/tf-keras-integration",
                   api_token="ANONYMOUS")


# Create a NeptuneCallback instance
neptune_cbk = NeptuneCallback(run=run, base_namespace="metrics")


# Pass the callback to model.fit()
model.fit(x_train, y_train,
          epochs=5,
          batch_size=64,
          callbacks=[neptune_cbk])


# Stop the run
run.stop()
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting-started/getting-help#frequently-asked-questions)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
