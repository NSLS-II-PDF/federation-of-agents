import tensorflow as tf
import numpy as np
from pathlib import Path
from scipy import interpolate
from collections import namedtuple

TransformPair = namedtuple("TransformPair", ["forward", "inverse"])


class XCACompanion:
    def __init__(
        self, *, model_path, model_tth_min, model_tth_max, n_datapoints, **kwargs
    ):
        """

        Parameters
        ----------
        model_path: str, Path
        model_tth_min: float
            2-theta minimum value from model's training dataset
        model_tth_max: float
            2-theta maximum value from model's training dataset
        n_datapoints: int
            Number of data points in model's training dataset. Used to construct linspace.
        kwargs
        """
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        self.model = tf.keras.models.load_model(str(model_path))
        self.model_tth = np.linspace(model_tth_min, model_tth_max, n_datapoints)
        self.independent = None
        self.dependent = None

    def preprocess(self, tth, I):
        """
        Performs interpolation and normalization.

        Parameters
        ----------
        tth
        I: ndarray
            Intensity array shape (m, n_datapoints) or (m, n_datapoints, 1)

        Returns
        -------

        """
        if tth.shape != self.model_tth.shape and np.any(tth != self.model_tth):
            inter = interpolate.interp1d(
                tth, I, kind="quadratic", bounds_error=False, fill_value=(0.0, 0.0)
            )
            x = inter(self.model_tth)
        else:
            x = I

        x = (x - np.min(x, axis=1, keepdims=True)) / (
            np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
        )
        x = np.reshape(x, (-1, len(self.model_tth), 1))

    def tell(self, x, y):
        """Tell the agent about some data"""
        raise NotImplementedError

    def ask(self):
        """Ask the agent for some advice"""
        raise NotImplementedError

    def observe(self):
        """Allow the agent to summarize observations"""
        raise NotImplementedError

    def __len__(self):
        if self.dependent is None:
            return 0
        else:
            return self.dependent.shape[0]


def default_transform_factory():
    """
    Constructs simple transform that does nothing.
    Forward goes from scientific coordinates to beamline coordinates
    Reverse goes from beamline coordinates to scientific coordinates
    """
    return TransformPair(lambda x: x, lambda x: x)


