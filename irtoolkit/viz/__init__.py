from math import ceil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import utils


def scale_figsize(nrows, ncols, scale=1):
    figsize = plt.rcParams["figure.figsize"]
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    c, r = scale

    return (c * ncols * figsize[0], r * nrows * figsize[1])


def clean_axes_grid(axes, xlabel=None, ylabel=None):
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")

    if xlabel is not None:
        for ax in axes.ravel():
            ax.set_xlabel("")

        xaxes = axes[-1] if axes.ndim > 1 else axes
        for cidx, ax in enumerate(xaxes):
            if ax.has_data():
                ax.set_xlabel(xlabel)
            elif axes.ndim > 1 and axes.shape[0] > 2:
                axes[-2, cidx].set_xlabel(xlabel)

    if ylabel is not None:
        for ax in axes.ravel():
            ax.set_ylabel("")
        yaxes = axes[:, 0] if axes.ndim > 1 else axes
        for ax in yaxes.ravel():
            ax.set_ylabel(ylabel)

    return


def create_axes_grid(nitems, ncols, figsize=None, scale=1, **kwargs):
    nrows = ceil(nitems / ncols)
    if figsize is None:
        figsize = scale_figsize(nrows, ncols, scale)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    if ncols == 1 and nrows == 1:
        axes = np.array([[axes]])
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    assert axes.ndim == 2

    return fig, axes


def modify_legend(ax, **kwargs):
    """Modify a legend object after it has been created."""
    legend = ax.get_legend()
    if legend is None:
        return

    handles = legend.legendHandles
    ax.legend(handles=handles, **kwargs)
    return


def legend_handle(label, color, cmap="tab10", marker=None, markersize=10, **kwargs):
    cmap = plt.get_cmap(cmap)
    if isinstance(color, int):
        color = cmap(color)

    if marker is None:
        return Patch(color=color, label=label, **kwargs)

    return Line2D(
        np.array([0]),
        np.array([0]),
        color="white",
        marker=marker,
        markersize=markersize,
        markerfacecolor=color,
        label=label,
        **kwargs,
    )


def legend_from_lut(lut, order=None, **kwargs):
    if order is None:
        order = lut.keys()

    return [legend_handle(key, lut[key], **kwargs) for key in order]


def extract_signal(f, key="raw", wn_start=1500, wn_end=1700):
    wn = f.attrs["wavenumber"]
    start, end = np.argmax(wn > wn_start), np.argmax(wn > wn_end)
    signal = f[key][:, :, start:end].mean(2)

    return signal


def lineplot(xyz, wn, errorbar="pi", **kwargs):
    assert len(xyz.shape) == 3
    assert xyz.shape[2] == len(wn)
    long = pd.DataFrame(xyz.reshape(-1, xyz.shape[2]), columns=wn).melt()
    sns.lineplot(data=long, x="variable", y="value", errorbar=errorbar, **kwargs)
    return


def plot_sample(path, filters=None, axes=None):
    if filters is None:
        filters = [(1000, 1200), (1200, 1300), (1350, 1500), (1500, 1600), (1600, 1700)]

    if axes is None:
        _, axes = plt.subplots(
            1,
            len(filters),
            figsize=(6.8 * len(filters), 4.8 * 1),
            sharex=False,
            sharey=False,
        )

    with h5py.File(path, "r") as f:
        for ax, (start, stop) in zip(axes.ravel(), filters):
            signal = utils.average_signal(f, start=start, stop=stop)
            ax.imshow(signal)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title(f"({start}, {stop})")

    return axes
