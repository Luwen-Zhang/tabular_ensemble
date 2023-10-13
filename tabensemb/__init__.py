import os
import numpy as np
import warnings
import sys

np.int = int  # ``np.int`` is a deprecated alias for the builtin ``int``.

__root__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["data", "model", "trainer", "config", "utils"]

__version__ = "0.1"

setting = dict(
    # The random seed for numpy and pytorch (including cuda).
    random_seed=42,
    # If the memory of the system (gpu or cpu) is lower than 6 GiBs, set ``low_memory`` to True.
    # TODO: Enlarge bayes search space when low_memory is set to False.
    low_memory=True,
    verbose_per_epoch=20,
    # To save memory, turn test_with_no_grad to True. However, this operation will make
    # some models that need gradients within the loss function invalid.
    test_with_no_grad=True,
    # Debug mode might change behaviors of models. By default, epoch will be set to 2, n_calls to minimum, and
    # bayes_epoch to 1.
    debug_mode=False,
    # Default paths to configure trainers, data modules, and models.
    default_output_path="output",
    default_config_path="configs",
    default_data_path="data",
    # If False, raise an Exception if calculating metrics for predictions containing NaNs. If True, the metric will
    # be 100 instead.
    warn_nan_metric=True,
    # Model bases will check the task type before training. If set to True and the inferred task is not consistent with
    # the configuration, an exception will be raised.
    raise_inconsistent_inferred_task=False,
    # Set matplotlib.rc("text", usetex=True) if latex installation is found.
    matplotlib_usetex=False,
    # The upper boundary of the returned objective value from a bayesian optimization iteration. Any objective value
    # beyond this value will be clipped to it.
    bayes_loss_limit=1000,
)

if setting["debug_mode"]:
    warnings.warn("The debug mode is activated. Please confirm whether it is desired.")


def check_grad_in_loss():
    if setting["test_with_no_grad"]:
        return False
    return True


class Stream:
    def __init__(self, stream, path=None):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self.set_stream(stream)
        self.set_path(path)

    def write(self, message):
        self.stream.write(message)
        if self.path is not None:
            with open(self.path, "ab") as log:
                log.write(message.encode("utf-8"))

    def set_stream(self, stream):
        if stream == "stdout":
            self.stream = self._stdout
        elif stream == "stderr":
            self.stream = self._stderr
        else:
            self.stream = stream

    def set_path(self, path):
        self.path = path

    def close(self):
        self.stream.close()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


_original_stdout_stream = sys.stdout
_original_stderr_stream = sys.stderr
stdout_stream = Stream("stdout")
stderr_stream = Stream("stderr")

if not "pytest" in sys.modules:
    sys.stdout = stdout_stream
    sys.stderr = stderr_stream
