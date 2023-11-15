from irtoolkit import io, utils
import numpy as np
import seaborn as sns
import h5py
from matplotlib.patches import Rectangle
import pandas as pd


class SlideSection:
    def __init__(self, icol, irow, color="lightgrey", name=None, shape=None):
        self.icol = icol
        self.irow = irow
        self.color = color
        self.name = name

        self.shape = shape
        if shape is not None:
            dx, dy, _ = shape
            index = np.arange(dx * dy).reshape(dx, dy)
            self.index = index[self.irow, self.icol].ravel()
            self.bitmask = np.in1d(index, self.index)


    def make_patch(self, edgecolor=None, facecolor="none", linewidth=1, **kwargs):
        if edgecolor is None:
            edgecolor = self.color

        (c, r) = (self.icol.start, self.irow.start)
        dc = self.icol.stop - self.icol.start
        dr = self.irow.stop - self.irow.start

        patch = Rectangle(
            (c, r),
            dc,
            dr,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            **kwargs,
        )

        return patch

    def extract(self, f, key="image", wnrange=None):
        if wnrange is None:
            return f[key][self.irow, self.icol, :]

        start, stop = wnrange
        return utils.average_signal(f, key, self.irow, self.icol, start, stop)

    def mask(self, signal, flatten=False):
        assert signal.ndim == 3
        vals = signal[self.irow, self.icol, :]

        if flatten:
            return vals.reshape(-1, vals.shape[2])

        return vals

    def df(self, f, key, melt=False):
        wn = f.attrs["wavenumber"]
        assert len(f[key].shape) == 3
        assert f[key].shape[-1] == len(wn)

        foo = pd.DataFrame(
            self.extract(f, key).reshape(-1, len(wn)), columns=pd.Index(wn, name="wn")
        )

        if melt:
            return foo.melt(ignore_index=False).reset_index()

        return foo


def slide_1_region_J():
    palette = sns.color_palette()
    with h5py.File(io.path("slide-1-region-J")) as f:
        shape = f["image"].shape

    fg_internal = SlideSection(
        slice(1000, 1200),
        slice(400, 500),
        shape=shape,
        color=palette[0],
        name="foreground - internal",
    )

    fg_external = SlideSection(
        slice(1100, 1250),
        slice(700, 800),
        shape=shape,
        color=palette[1],
        name="foreground - external",
    )

    bg = SlideSection(
        slice(400, 600),
        slice(700, 800),
        shape=shape,
        color=palette[2],
        name="background",
    )

    regions = [fg_internal, fg_external, bg]

    return regions


def get_regions(sample):
    if sample == "slide-1-region-J":
        return slide_1_region_J()

    return
