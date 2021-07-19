import tensorflow as tf
import numpy as np
from pathlib import Path
from scipy import interpolate


class XCACompanion:
    def __init__(self, *, model_path, model_tth, **kwargs):
        """

        Parameters
        ----------
        model_path: str, Path
        model_tth_: array
            Model 2-theta linspace
        kwargs
        """
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        self.model = tf.keras.models.load_model(str(model_path))
        self.model_tth = model_tth
        self.independent = None
        self.dependent = None

    def preprocess(self, tth, intensity):
        """
        Performs interpolation and normalization.

        Parameters
        ----------
        tth
        intensity: ndarray
            Intensity array shape (m, n_datapoints) or (m, n_datapoints, 1)

        Returns
        -------

        """
        if tth.shape != self.model_tth.shape and np.any(tth != self.model_tth):
            inter = interpolate.interp1d(
                tth,
                intensity,
                kind="quadratic",
                bounds_error=False,
                fill_value=(0.0, 0.0),
            )
            x = inter(self.model_tth)
        else:
            x = intensity

        x = (
            (x - np.min(x, axis=1, keepdims=True))
            / (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))
        ) * 2 - 1
        x = np.reshape(x, (-1, len(self.model_tth), 1))
        return x

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
