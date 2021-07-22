import numpy as np
from scipy import interpolate


class XCACompanion:
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        model_path: str, Path
        model_tth_: array
            Model 2-theta linspace
        kwargs
        """
        self.model_name = None
        self.independent = None
        self.dependent = None

    def preprocess(self, tth, intensity):
        """
        Performs interpolation and normalization onto (-1,1).

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
