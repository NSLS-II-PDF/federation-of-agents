import tensorflow as tf
import numpy as np
from federation.xca import XCACompanion, default_transform_factory
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython import display


class CNNCompanion(XCACompanion):
    def __init__(
        self,
        *,
        model_path,
        model_tth,
        exp_tth,
        categorical=True,
        coordinate_transform=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        model_path: str, Path
        model_tth_: array
            Model 2-theta linspace
        exp_tth: array
            Experimental 2-theta linspace
        categorical: bool
            True for classification models, False for regression models
        coordinate_transform: Callable
            Optional transformation for independent variables in tell.
            Useful for converting "scientific" space coordinates to less interpretable or reduced
            "beamline" space coordinates.
        kwargs
        """
        super().__init__(model_path=model_path, model_tth=model_tth, **kwargs)
        self.exp_tth = exp_tth
        self.categorical = categorical
        if coordinate_transform is None:
            self.coordinate_transform = default_transform_factory()

    def predict(self, I):
        X = self.preprocess(self.exp_tth, I)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y_preds = self.model(X, training=False)
        return [y_preds[i, :] for i in range(y_preds.shape[0])]

    @staticmethod
    def entropy(y_preds):
        # Maximum entropy is Sum((1/n_classes)*log2(1/n_classes)) = 2.0 for 4 classes.
        H = np.sum(-y_preds * np.log2(y_preds + 1e-16), axis=-1)
        return H

    def tell(self, x, y):
        """
        Tell XCA about something new
        Parameters
        ----------
        x: These are the interesting parameters
        y: This should be the I(Q) shape (1, n_datapoints)

        Returns
        -------
        """
        ys = np.reshape(y, (1, -1))
        xs = np.reshape(x, (1, -1))
        self.tell_many(xs, ys)

    def tell_many(self, xs, ys):
        """
        Tell XCA about many new things
        Parameters
        ----------
        xs: These are the interesting parameters, they get converted to  space via a transform
        ys: list, arr
            This should be a list length m of the Q/I(Q) shape (n, 2)

        Returns
        -------

        """
        new_independents = list()
        for i in range(xs.shape[0]):
            new_independents.append(self.coordinate_transform.forward(*xs[i, :]))
        X = np.array(ys)
        y_preds = self.predict(X)
        y_preds = np.array(y_preds)
        if self.independent is None:
            self.independent = np.array(new_independents)
            self.dependent = y_preds
        else:
            self.independent = np.vstack([self.independent, new_independents])
            self.dependent = np.vstack([self.dependent, y_preds])

    def ask(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def record_output_dependents(self, path):
        """
        Saves the output dependent variables from the model and experiment
        Parameters
        ----------
        path: str, Path
            Output path for dataset

        Returns
        -------

        """
        from pandas import DataFrame

        data_dict = {}
        # Transformed coordinates held in object
        for i in range(self.independent.shape[1]):
            data_dict[f"Transformed Independent Variable {i}"] = self.independent[:, i]

        # Scientific coordinates
        reversed_independent = np.array(
            [
                self.coordinate_transform.reverse(*self.independent[i, :])
                for i in range(len(self))
            ]
        )
        for i in range(reversed_independent.shape[1]):
            data_dict[f"Independent Variable {i}"] = self.independent[:, i]

        for i in range(self.dependent.shape[1]):
            data_dict[f"Independent Variable {i}"] = self.dependent[:, i]

        df = DataFrame.from_dict(data_dict)
        df.to_csv(path)
        return df


class PlotCompanion(CNNCompanion):
    def __init__(self, model_path, model_tth, exp_tth, ax=None, **kwargs):
        """

        Parameters
        ----------
        model_path: str, Path
        model_tth_: array
            Model 2-theta linspace
        exp_tth: array
            Experimental 2-theta linspace
        ax: axis
            Plotting axis optional
        kwargs
        """
        super().__init__(
            model_path=model_path, model_tth=model_tth, exp_tth=exp_tth, **kwargs
        )
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = ax.figure

    def categorical_plot(self, y):
        """Creates a bar plot of point of interest"""
        height = np.ravel(y)
        x = np.arange(1, len(height) + 1)
        self.ax.bar(x, y)
        self.ax.set_xticks(x)
        self.ax.set_ylim([0, 1])

    def continuous_plot(self):
        """Creates a line plot of the dataset with labels"""
        idx = self.independent.argsort()
        self.ax.plot(self.dependent[idx])
        self.ax.set_yticks(np.arange(1, len(self) + 1))
        self.ax.set_yticklabels(self.independent[idx, :])

    def update_plot(self, independent=None):
        """

        Parameters
        ----------
        independent: ndarray
            Optional independent variable to choose from for categortical

        Returns
        -------

        """
        # Clear plot
        self.ax.cla()

        # Produce appropriate classification probability or regression task
        if self.categorical:
            if independent is None:
                ys = self.dependent[-1, :]
            else:
                idx = np.argwhere(
                    self.coordinate_transform.forward(*independent) == self.independent
                )
                ys = self.dependent[idx, :]
            self.categorical_plot(ys)
        else:
            self.continuous_plot()

        # Polish the rest off
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        display.clear_output(wait=True)
        display.display(self.fig)

    def observe(self, *args, **kwargs):
        self.update_plot(**kwargs)


class SearchCompanion(CNNCompanion):
    def __init__(self):
        super().__init__()
        self.cache = set()  # Hashable cache of proposals
        self.proposals = list()  # More data rich list of proposals
        raise NotImplementedError
