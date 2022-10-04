#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import io
import tempfile

# Note: we purposefully try to import `tensorflow.keras.callbacks.Callback`
# before `keras.callbacks.Callback` because the former is compatible with both
# `tensorflow.keras` and `keras`, while the latter is only compatible
# with `keras`. See https://github.com/keras-team/keras/issues/14125

try:
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.utils import model_to_dot
except ImportError:
    try:
        from keras.callbacks import Callback
        from keras.utils import model_to_dot
    except ImportError:
        msg = """
        keras package not found.

        As Keras is now part of Tensorflow you should install it by running
            pip install tensorflow"""
        raise ModuleNotFoundError(msg)  # pylint:disable=undefined-variable

try:
    # neptune-client=0.9.0+ package structure
    from neptune.new import Run
    from neptune.new.exceptions import NeptuneException
    from neptune.new.integrations.utils import expect_not_an_experiment, verify_type
    from neptune.new.types import File
except ImportError:
    # neptune-client>=1.0.0 package structure
    from neptune import Run
    from neptune.exceptions import NeptuneException
    from neptune.integrations.utils import verify_type, expect_not_an_experiment
    from neptune.types import File

from neptune_tensorflow_keras import __version__

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-tensorflow-keras"


class NeptuneCallback(Callback):
    """Captures model training metadata and logs them to Neptune.

    See the example run here https://ui.neptune.ai/shared/keras-integration/e/KERAS-23/logs

    Args:
        run (`neptune.new.Run`): Neptune run.
        base_namespace (`str`, optional): Namespace (folder) under which all metadata
            logged by the NeptuneCallback will be stored. Defaults to "training".
        log_on_batch (`bool`): Log the metrics also for each batch, not only each epoch.
        log_model_diagram (`bool`): Save the model visualization. Defaults to False.
            Requires pydot to be installed, otherwise it will silently skip saving the diagram.

    Example:

        Initialize Neptune client:

        >>> import neptune.new as neptune
        >>> run = neptune.init_run(
        ...     project="common/tf-keras-integration",
        ...     api_token=neptune.ANONYMOUS_API_TOKEN,
        ... )

        Instantiate the callback and pass it to the `callbacks` argument of `model.fit()`:

        >>> from neptune.new.integrations.tensorflow_keras import NeptuneCallback
        >>> neptune_callback = NeptuneCallback(run=run)
        >>> model.fit(x_train, y_train, callbacks=[neptune_callback])

    Note:
        To use this module, you need to have Keras or Tensorflow 2 installed.
    """

    def __init__(
        self,
        run: Run,
        base_namespace: str = "training",
        log_model_diagram: bool = False,
        log_on_batch: bool = False,
    ):
        super().__init__()

        expect_not_an_experiment(run)
        verify_type("run", run, Run)
        verify_type("base_namespace", base_namespace, (str, type(None)))
        verify_type("log_model_diagram", log_model_diagram, bool)

        self._run = run
        self._log_model_diagram = log_model_diagram

        self._log_on_batch = log_on_batch

        if base_namespace.endswith("/"):
            self._base_namespace = base_namespace[:-1]
        else:
            self._base_namespace = base_namespace

        self._run[INTEGRATION_VERSION_KEY] = __version__

    @property
    def _metric_logger(self):
        return self._run[self._base_namespace]

    @property
    def _model_logger(self):
        return self._run[self._base_namespace]["model"]

    def _log_metrics(self, logs, category: str, trigger: str):
        if not logs:
            return

        logger = self._metric_logger[category][trigger]

        for metric, value in logs.items():
            try:
                if metric in ("batch", "size") or metric.startswith("val_"):
                    continue
                logger[metric].log(value)
            except NeptuneException:
                pass

    def on_train_begin(self, logs=None):  # pylint:disable=unused-argument
        self._model_logger["optimizer_config"] = self.model.optimizer.get_config()  # it is a dict
        self._metric_logger["fit_params"] = self.params

    def on_train_end(self, logs=None):  # pylint:disable=unused-argument
        # We need this to be logged at the end of the training, otherwise we are risking this to happen:
        # https://stackoverflow.com/q/55908188/3986320
        self._model_logger["summary"] = _model_summary_file(self.model)

        if self._log_model_diagram:
            self._model_logger["visualization"] = _model_diagram(self.model)

    def on_train_batch_end(self, batch, logs=None):  # pylint:disable=unused-argument
        if self._log_on_batch:
            self._log_metrics(logs, "train", "batch")

    def on_epoch_begin(self, epoch, logs=None):  # pylint:disable=unused-argument
        self._model_logger["learning_rate"].log(self.model.optimizer.learning_rate)

    def on_epoch_end(self, epoch, logs=None):  # pylint:disable=unused-argument
        self._log_metrics(logs, "train", "epoch")

    def on_test_batch_end(self, batch, logs=None):  # pylint:disable=unused-argument
        if self._log_on_batch:
            self._log_metrics(logs, "validation", "batch")

    def on_test_end(self, logs=None):  # pylint:disable=unused-argument
        self._log_metrics(logs, "validation", "epoch")


def _model_summary_file(model) -> File:
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return File.from_stream(stream, extension="txt")


def _model_diagram(model) -> File:
    dot = model_to_dot(model)
    if dot is not None:
        # the same as TF/Keras does, we will fail with ImportError unless using a notebook,
        # where it just prints a warning message
        tmp = tempfile.NamedTemporaryFile(delete=False)
        dot.write(tmp.name, format="png")
        return File(tmp.name, extension="png")
