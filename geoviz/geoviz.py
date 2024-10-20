"""
Geoviz Module for the geometric visualization of transformations inside 
a feedforward neural network.
Author: Francesco Pierpaoli
Email: francescopierpaoli96@gmail.com
"""
import warnings
from math import ceil, floor
from typing import List, Tuple, Any
from collections.abc import Callable
from copy import deepcopy
from cv2 import VideoWriter, VideoWriter_fourcc, destroyAllWindows
import numpy as np
from matplotlib import pyplot as plt, cm
import matplotlib.figure
from moviepy.video.io.bindings import mplfig_to_npimage
from mlxtend.plotting import plot_decision_regions
import torch
import torch.nn as nn
from highdimensional_boundary import DBPlot
from mpl_toolkits.mplot3d import Axes3D
import imageio.v3 as iio

colormap = cm.get_cmap("rainbow")

def plot_grid_func(
    n_lines: int,
    line_points: int,
    ax: plt.Axes,
    map_func: Callable[[torch.Tensor], torch.Tensor] = nn.Identity(),
    starting_grid: List[np.ndarray] | None = None,
    dim: int = 2,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> List[np.ndarray]:
    """
    Function which plots on ax the transformation of starting grid via map_func 
    and returns the transformed grid.

    Args:
        n_lines (int): number of lines per axis in the grid.
        line_points (int): number of points sampled for each line.
        ax (plt.Axes): matplolib axes to plot the grid on.
        map_func (Callable[[torch.Tensor], torch.Tensor], optional): Transformation for starting_grid. 
        Defaults to nn.Identity().
        starting_grid (List[np.ndarray] | None, optional): Starting grid to transform. 
        If None, it is created a standard regular grid. Defaults to None.
        dim (int, optional): Dimension of the euclidean space for the grid, can be 2 or 3. 
        Defaults to 2.
        x_min (float | None, optional): Minimum x value for build the regular grid. Defaults to None.
        x_max (float | None, optional): Maximum x value for build the regular grid. Defaults to None.
        y_min (float | None, optional): Minimum y value for build the regular grid. Defaults to None.
        y_max (float | None, optional): Maximum y value for build the regular grid. Defaults to None.

    Returns:
        List[np.ndarray]: Transformed grid via map_func.
    """
    # List for gathering the lines into.
    lines = []

    if starting_grid is None:
        # Iterate over horizontal lines.
        for y in np.linspace(y_min, y_max, n_lines):
            lines.append(
                np.array([[x, y] for x in np.linspace(x_min, x_max, line_points)])
            )

        # Iterate over vertical lines.
        for x in np.linspace(x_min, x_max, n_lines):
            lines.append(
                np.array([[x, y] for y in np.linspace(y_min, y_max, line_points)])
            )

    else:
        for line in starting_grid:
            with torch.no_grad():
                transformed_line = map_func(torch.Tensor(line)).numpy()
            lines.append(transformed_line)

    # adapt this if we allow starting 3d data with corresponding 3d axis grid
    step = int(len(lines) / (n_lines * 2))
    to_keep = [lines[floor(len(lines) / 2)], lines[ceil(len(lines) / 2)], lines[-1]]

    for i, line in enumerate(lines[::step] + to_keep):
        i = i * step
        if dim == 2:
            p = i / (len(lines) - 1)  # Normalize to 0-1.
            # Get the line color from the colormap.
            ax.plot(line[:, 0], line[:, 1], color=colormap(p), linewidth=0.7)
        elif dim == 3:
            p = i / (len(lines) - 1)
            ax.plot3D(
                line[:, 0], line[:, 1], line[:, 2], color=colormap(p), linewidth=0.7
            )
        else:
            raise ValueError("The only dimensions allowed are 2 or 3.")

    return lines


class VizModule(nn.Module):
    """ Subclass of pytorch base nn.Module to enrich it with necessary methods for the library. 
    -------
    Constructors: 

    __init__

    Methods:

    register_transformations()
    viz_udpate()
    neurons_count()
    """

    def __init__(self):
        super().__init__()
        self.transformations = {}

    def register_transformations(self, epoch):
        """
        Method to register all the transformations during the model training. TODO: avoid deepcopy. 
        """
        self.transformations[epoch] = {}

        with torch.no_grad():
            for layer, module in self.named_children():
                self.transformations[epoch][layer] = deepcopy(module)

    def viz_update(self, epoch):
        """
        Alias for register_transformations
        """
        self.register_transformations(epoch)

    def neurons_count(self):
        """
        Helper method for counting neurons inside each layer of the model. 
        """
        res = {}
        for id, (layer, module) in enumerate(self.named_children()):
            if isinstance(module, nn.Linear):
                if id == 0:
                    res["input"] = module.in_features
                res[layer] = module.out_features
        return res


class geo_viz:
    """
    Base class for the geometric visualization of classification neural networks during training. 
    -------
    Constructors: 

    __init__

    Methods:

    plot()
    save_image()
    make_gif()
    make_video()
    predict()
    predict_proba()
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: VizModule,
        DR: Any = None,
        estimate_decision_boundary: bool = True,
    ):
        """Constructor. 

        Args:
            X_train (np.ndarray): Train set for the model.
            y_train (np.ndarray): Label set for the model.
            model (geoviz.VizModule): Trained instance of the VizModule class.
            DR (Any): Dimensionality reduction object for visualizing layers and/or decision boundary in dim>3. 
            It must have both .fit and .transform methods. 
        """

        self.transformations = model.transformations
        self.n_epochs = len(self.transformations)
        self.epoch = None
        self.annotate = None
        self.plot_grid = None
        self.DR = DR
        self.estimate_decision_boundary = estimate_decision_boundary

        self.X_train, self.y_train = (
            self.__validate_array(X_train),
            self.__validate_array(y_train),
        )

        for idx, (layer, n_neurons) in enumerate(model.neurons_count().items()):
            if n_neurons > 3:
                if self.DR is None:
                    warnings.warn(
                        """Some layers of the model have more than 3 dims. 
                                  You must also pass a dimensionality reduction object with 
                                  a transform method to plot transformations for these layers,
                                  otherwise the plots will be empty. 
                                  Use argument DR in the geoviz object definition. 
                                  """
                    )

        self.n_layers = len(model.neurons_count())
        if DR is not None:  # fit on input points if necessary
            try:
                self.DR.transform([self.X_train[0]])
            except:
                self.DR.fit(self.X_train)

    @staticmethod
    def __validate_array(X):
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X)
            except Exception as exc:
                raise Exception(
                    "Impossible to cast X_train and/or y_train to numpy array."
                ) from exc
        return X

    def plot(
        self,
        epoch: int,
        figsize: Tuple[int, int] | None = None,
        annotate: float | None = None,
        plot_grid: bool = False,
        border: float = 0.05,
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Method to return for a specific epoch of the model training a matplotlib figure with: 
            1. The input points of training set
            2. The decision boundary of the model
            3. All the transformations inside the model

        Args:
            epoch (int): The epoch.
            figsize (Tuple[int,int] | None, optional): plt.figure argument figsize for the figure size.
            annotate (float | None, optional): Percentage of points in the training set to label. If None there
            will be no annotations. Defaults to None.
            plot_grid (bool, optional): Whether or not to also plot a grid during the transformations.
            At the moment, it only plots for input points in 2 dims and layer transformations <=3. 
            Defaults to False.
            border (float , optional): If also plotting the grid, percentage of margin wrt the points to give to the starting grid.
            Defaults to 0.05.
            **kwargs: matplotlib kwargs in plotting functions (example: alpha, colors, ecc...).

        Returns:
            plt.fig: The matplolib figure.
        """
        kwargs.setdefault("markersize", 5)
        kwargs.setdefault("alpha", 0.75)
        self.epoch = epoch
        self.annotate = annotate
        self.plot_grid = plot_grid
        classes = np.unique(self.y_train)
        markers = ["o", "<", "x"]  # extend this list
        dim = self.X_train.shape[1]  # put this in a better place

        # Initialize matplotlib fig based on layer number
        fig = plt.figure(figsize=figsize, num=epoch, clear=True)
        nrows = self.n_layers
        ncols = 2  # taking into account activations output

        # Input points
        if dim == 2:
            ax = fig.add_subplot(nrows, ncols, 1)
            title = "Input points"
        elif dim == 3:
            ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
            title = "Input points"
        else:
            ax = fig.add_subplot(nrows, ncols, 1)
            if self.DR is not None:
                X_train_embedded = self.DR.transform(self.X_train)
                title = f"Input points [Projected with {self.DR.__class__.__name__}]"
            else:
                title = "Input points [No projection provided]"

        for i, class_ in enumerate(classes):
            if dim > 3:
                if self.DR is not None:
                    input = X_train_embedded[self.y_train == class_, :]
            else:
                input = self.X_train[self.y_train == class_, :]

            if dim == 2:
                ax.plot(input[:, 0], input[:, 1], markers[i], **kwargs)
            elif dim == 3:
                ax.scatter(
                    input[:, 0], input[:, 1], input[:, 2], markers[i], alpha=0.75, s=5
                )
            else:
                if self.DR is not None:
                    ax.plot(input[:, 0], input[:, 1], markers[i], **kwargs)

        if self.annotate:
            input = self.X_train
            step = int(len(self.X_train) * self.annotate) + 1  # improve this
            for i, txt in enumerate(range(self.X_train.shape[0])[:step:]):
                if dim == 2:
                    ax.annotate(txt, (self.X_train[i, 0], self.X_train[i, 1]))
                elif dim == 3:
                    ax.text(
                        input[i, 0],
                        input[i, 1],
                        input[i, 2],
                        txt,
                        zdir="x",
                        fontsize="xx-small",
                    )
                else:
                    if self.DR is not None:
                        ax.annotate(
                            txt, (X_train_embedded[i, 0], X_train_embedded[i, 1])
                        )

        if (
            self.plot_grid
        ):  # adapt plot grid to use pytorch model.transformations functions
            if dim >= 3: #TODO: Implement for dim>=3
                pass
            else:
                x_min_, x_max_ = np.min(self.X_train[:, 0]), np.max(self.X_train[:, 0])
                y_min_, y_max_ = np.min(self.X_train[:, 1]), np.max(self.X_train[:, 1])
                x_min = x_min_ - abs(x_max_ - x_min_) * border
                x_max = x_max_ + abs(x_max_ - x_min_) * border
                y_min = y_min_ - abs(y_max_ - y_min_) * border
                y_max = y_max_ + abs(y_max_ - y_min_) * border

            if dim == 3:
               pass

            if dim == 2:
                lines_identity = plot_grid_func(
                    10, 10, ax, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
                )
                tmp_grid = lines_identity
            elif dim == 3:
                pass

        ax.set_title(title)

        # Decision boundary plot
        ax = fig.add_subplot(nrows, ncols, 2)
        if dim == 2:
            input = self.X_train
            plot_decision_regions(input, self.y_train.astype(int), self)
            title = "Decision boundary"
        else:
            if (self.DR is not None) and (self.estimate_decision_boundary):
                warnings.warn(
                    """Estimating the decision boundary is done via the experimental library
                    highdimensional_boundary, and it may takes a while. 
                    """)
                db = DBPlot(
                    self, self.DR, verbose=False, n_decision_boundary_keypoints=45
                )  # Argument n_decision_boundary_keypoints configurable?
                db.fit(self.X_train, self.y_train.astype(int))
                db.plot(
                    ax,
                    generate_testpoints=True,
                    legend=False,
                    generate_background=False,
                    annotate=self.annotate,
                    scatter_size_scale=0.5,
                )
                title = f"Decision boundary [Estimated via projection with {self.DR.__class__.__name__}]"
            else:
                title = "Decision boundary [No projection provided or estimation requested]"
        ax.set_title(title)

        # Layers and activations
        output = torch.Tensor(self.X_train)
        for id, (key, transformation) in enumerate(self.transformations[epoch].items()):

            output = transformation(output).data
            n_img = id + 3
            dim = output.shape[1]  # output neurons

            if dim in (1,2):
                ax = fig.add_subplot(nrows, ncols, n_img)
                title = ""
            elif dim == 3:
                ax = fig.add_subplot(nrows, ncols, n_img, projection="3d")
                title = ""
            else:
                ax = fig.add_subplot(nrows, ncols, n_img)
                if self.DR is not None:
                    fitted_DR = self.DR.fit(output)
                    out_embedded = fitted_DR.transform(output)
                    title = f"[Projected with {self.DR.__class__.__name__}]"
                else:
                    title = "[No projection provided]"

            for i, class_ in enumerate(classes):
                if dim > 3:
                    if self.DR is not None:
                        out = out_embedded[self.y_train == class_, :]
                else:
                    out = output.numpy()[self.y_train == class_, :]

                if dim == 1:
                    ax.plot(out[:, 0], np.zeros_like(out) + 0, markers[i], **kwargs)
                elif dim == 2:
                    # adapt with input vector y_label
                    ax.plot(out[:, 0], out[:, 1], markers[i], **kwargs)
                elif dim == 3:
                    ax.scatter(
                        out[:, 0],
                        out[:, 1],
                        out[:, 2],
                        marker=markers[i],
                        alpha=0.75,
                        s=5,
                    )
                else:
                    if self.DR is not None:
                        ax.plot(out[:, 0], out[:, 1], markers[i], **kwargs)

            if self.annotate:
                step = int(len(self.X_train) * self.annotate) + 1  # improve this
                for i, txt in enumerate(range(self.X_train.shape[0])[:step:]):
                    if dim == 1:
                        ax.annotate(txt, (output.numpy()[i], 0))
                    elif dim == 2:
                        ax.annotate(txt, (output.numpy()[i, 0], output.numpy()[i, 1]))
                    elif dim == 3:
                        ax.text(
                            output.numpy()[i, 0],
                            output.numpy()[i, 1],
                            output.numpy()[i, 2],
                            txt,
                            zdir="x",
                            fontsize="xx-small",
                        )
                    else:
                        if self.DR is not None:
                            ax.annotate(txt, (out_embedded[i, 0], out_embedded[i, 1]))

            if self.plot_grid:
                if dim == 1:
                    # ax.plot(out[:,0],np.zeros_like(out) + 0, markers[i], alpha = 0.75, markersize= 5) ## adapt with input vector y_label
                    pass
                elif dim == 2:
                    tmp_grid = plot_grid_func(
                        10, 10, ax, map_func=transformation, starting_grid=tmp_grid
                    )
                elif dim == 3:
                    tmp_grid = plot_grid_func(
                        10,
                        10,
                        ax,
                        map_func=transformation,
                        starting_grid=tmp_grid,
                        dim=3,
                    )
                else: #TODO: implement for dim>3
                    pass
                # lines_identity = plot_grid(x_min,x_max,y_min,y_max,10,10,ax, map_func = None)

            ax.set_title(key + " " + title)
            fig.suptitle(f"Epoch {epoch}")
            fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def save_image(self, path_out: str, epoch: int, **kwargs) -> None:
        """Method to create image via self.plot and save it to disk. 

        Args:
            path_out (str): path to save the fig to. 
            epoch (int): epoch. 
            **kwargs: keyword arguments to pass to self.plot().
        """

        img = self.plot(epoch=epoch, **kwargs)
        img.savefig(path_out)
        img.clf()
        plt.close(img)

    def make_gif(
        self,
        path_out: str,
        step: int | None = None,
        duration: int = 100,
        loop: int = 0,
        **kwargs,
    ) -> None:
        """ Method to create and save a gif with imageio library from self.plot images every **step** epochs.

        Args:
            path_out (str): path to save the gif to. 
            step (int | None, optional): every how many epochs starting from 0 plot the images.
            If None, it is taken as int(n_epochs/10). Defaults to None.
            duration (int, optional): imageio argument duration for gif duration (in milliseconds). Defaults to 100.
            loop (int, optional): imageio argument loop. Defaults to 0.
            **kwargs: keyword arguments to pass to self.plot().
        """
        if step is None:
            step = int(self.n_epochs / 10)
        images = []
        for epoch in range(self.n_epochs):
            if epoch % step == 0 or epoch + 1 == self.n_epochs:
                fig = self.plot(epoch, **kwargs)
                images.append(mplfig_to_npimage(fig))
                fig.clf()
                plt.close(fig)
        iio.imwrite(path_out, images, duration=duration, loop=loop)

    def make_video(
        self,
        fps=1,
        title="video_out.avi",
        step=None,
        figsize=None,
        annotate=False,
        plot_grid=False,
    ):

        if step is None:
            step = int(self.n_epochs / 10)

        for epoch in range(self.n_epochs):
            if epoch % step == 0 or epoch + 1 == self.n_epochs:
                img = self.plot(
                    epoch, figsize=figsize, annotate=annotate, plot_grid=plot_grid
                )
                if epoch == 0:
                    figsize_int = img.get_size_inches() * img.dpi
                    out = VideoWriter(
                        title,
                        VideoWriter_fourcc(*"DIVX"),
                        fps,
                        tuple(figsize_int.astype(int)),
                    )
                out.write(mplfig_to_npimage(img))

                # closing figure
                img.clf()
                plt.close(img)

        out.release()
        destroyAllWindows()

    def predict(self, x):
        """
        Method to mimic pytorch nn.Module method in order to use the mlxtend.plot_decision_regions method.
        """

        x = torch.tensor(x, dtype=torch.float32)
        for trasf in self.transformations[self.epoch].values():
            x = trasf(x)

        pred = x[:, 0]
        return (pred >= 0.5).detach().numpy().astype(int)

    def predict_proba(self, x):
        """
        Method to use the highdimensional-decision-boundary-plot library.
        """
        x = torch.tensor(x, dtype=torch.float32)

        for trasf in self.transformations[self.epoch].values():
            x = trasf(x)

        x = torch.cat((1 - abs(x), x), dim=1)
        return x.detach().numpy()
