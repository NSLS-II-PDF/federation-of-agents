from deepnmf.companion.nmf import decomposition, iterative_decomposition
from federation.utils.transforms import default_transform_factory
from federation.utils.plotting import waterfall
import numpy as np
from matplotlib.pyplot import figure
import matplotlib as mpl
from IPython import display


class NMFCompanion:
    def __init__(
        self,
        n_components,
        *,
        q,
        coordinate_transform=None,
        deconvolutional=False,
        kernel_width=None,
        fixed_components=None,
        normalize=True,
        fig=None,
        cmap="tab10",
    ):
        """
        Base class for NMF companion agent.
        Parameters
        ----------
        n_components: int
            Number of components for NMF
        q: array
            Q space for measurement
        coordinate_transform: Callable
            Optional transformation for independent variables in tell.
            Useful for converting "scientific" space coordinates to less interpretable or reduced
            "beamline" space coordinates.
        deconvolutional: bool
            Operational mode for NMF.
        kernel_width: int
            Width of 1-dimensional convolutional kernel, required if deconvolutional is True.
        fixed_components: None, array
            Initial fixed components for NMF decomposition
        normalize: bool
            Normalize data in decomposition
        fig: Figure
        cmap: str
            Matplotlib colormap


        Returns
        -------

        """
        self.n_components = n_components
        self.q = q
        self.independent = None
        self.dependent_components = None  # NMF Components
        self.dependent_weights = None  # NMF Weights
        self.dependent = None  # Raw Data
        if coordinate_transform is None:
            self.coordinate_transform = default_transform_factory()
        else:
            self.coordinate_transform = coordinate_transform
        self.deconvolutional = deconvolutional
        self.fixed_components = fixed_components
        self.normalize = normalize
        if fig is None:
            self.fig = figure()
        else:
            self.fig = fig
        axes = self.fig.subplots(2, 2)
        self.component_ax = axes[0, 0]
        self.weight_ax = axes[0, 1]
        self.loss_ax = axes[1, 0]
        self.residual_ax = axes[1, 1]
        self.plot_order = list(range(n_components))  # Order for plotting
        self.cmap = mpl.cm.get_cmap(cmap)
        self.norm = mpl.colors.Normalize(vmin=0, vmax=n_components)

        if self.deconvolutional and kernel_width is None:
            raise ValueError(
                "kernel_width is a required argument for NMFCompanion when deconvolutional mode is used."
            )

    def update_decomposition(self):
        if self.deconvolutional:
            mode = "Deconvolutional"
        else:
            mode = "Linear"
        _, _, self.dependent_weights, self.dependent_components = decomposition(
            np.arange(self.dependent.shape[1]),
            self.dependent,
            n_components=self.n_components,
            initial_components=self.fixed_components,
            fix_components=[True for _ in range(len(self.fixed_components))],
            mode=mode,
            normalize=self.normalize,
        )

    def tell(self, x, y):
        """
        Tell the NMF about something new
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
        Tell the NMF about many new things
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
        new_dependents = np.array(ys)
        if self.independent is None:
            self.independent = np.array(new_independents)
            self.dependent = input
        else:
            self.independent = np.vstack([self.independent, new_independents])
            self.dependent = np.vstack([self.dependent, new_dependents])
        self.update_decomposition()

    def update_plot_order(self):
        """
        Order by proxy center of mass of class in plot regime.
        Makes the plots feel like a progression not random.
        """
        self.plot_order = np.argsort(
            np.matmul(
                np.arange(self.dependent_weights.shape[0]), self.dependent_weights
            )
        )

    def update_weights_plot(self):
        self.weight_ax.cla()
        for i in range(self.dependent_weights.shape[1]):
            self.weight_ax.plot(
                self.independent,
                self.dependent_weights[:, self.plot_order[i]],
                color=self.cmap(self.norm(i)),
                label=f"Component {i + 1}",
            )

    def update_loss_plot(self):
        self.loss_ax.cla()
        WH = np.matmul(self.dependent_weights, self.dependent_components)
        loss = np.mean((WH - self.dependent) ** 2, axis=1)
        self.loss_ax.plot(self.independent, loss)

    def update_component_plot(self):
        self.component_ax.cla()
        kernel_width = len(self.q) - self.dependent_components.shape[1] + 1
        for i in range(self.dependent_components.shape[0]):
            if kernel_width == 1:
                self.component_ax.plot(
                    self.q,
                    self.dependent_components[self.plot_order[i], :] + i,
                    color=self.cmap(self.norm(i)),
                )
            else:
                start_idx = kernel_width // 2
                finish_index = -kernel_width // 2 + 1
                self.component_ax.plot(
                    self.q[start_idx:finish_index],
                    self.dependent_components[self.plot_order[i], :] + i,
                    color=self.cmap(self.norm(i)),
                )

    def update_residual_plot(self):
        self.residual_ax.cla()
        residuals = (
            np.matmul(self.dependent_weights, self.dependent_components)
            - self.dependent
        )
        waterfall(
            self.residual_ax, self.q, residuals, alphas=np.mean(residuals, axis=1)
        )

    def ask(self):
        """Ask the agent for some advice"""
        raise NotImplementedError

    def observe(self, **kwargs):
        """Allow the agent to summarize observations"""
        self.update_plot_order()
        self.update_weights_plot()
        self.update_component_plot()
        self.update_loss_plot()
        self.update_residual_plot()

        # Polish the rest off
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        display.clear_output(wait=True)
        display.display(self.fig)

    def __len__(self):
        if self.dependent is None:
            return 0
        else:
            return self.dependent.shape[0]


class AutoNMFCompanion(NMFCompanion):
    def __init__(self, n_components, *, q, **kwargs):
        super().__init__(n_components, q=q, **kwargs)

    def update_decomposition(self):
        if self.deconvolutional:
            mode = "Deconvolutional"
        else:
            mode = "Linear"
        (
            _,
            _,
            self.dependent_weights,
            self.dependent_components,
        ) = iterative_decomposition(
            np.arange(self.dependent.shape[1]),
            self.dependent,
            n_components=self.n_components,
            mode=mode,
            normalize=self.normalize,
        )
