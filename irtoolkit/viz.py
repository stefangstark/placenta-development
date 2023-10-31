import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from matplotlib.patches import Rectangle


def extract_signal(f, key='raw', wn_start=1500, wn_end=1700):
    
    wn = f.attrs['wavenumber']
    start, end = np.argmax(wn > wn_start), np.argmax(wn > wn_end)
    signal = f[key][:, :, start:end].mean(2)

    return signal


def lineplot(xyz, errorbar='pi', **kwargs):
    assert len(xyz.shape) == 3
    long = pd.DataFrame(xyz.reshape(-1, xyz.shape[2])).melt()
    sns.lineplot(data=long, x='variable', y='value', errorbar=errorbar, **kwargs)
    return


@dataclass
class SlideSection:
    icol: ...
    irow: ...
    color: ... = 'lightgrey'
    name: ... = None

    def make_patch(self, edgecolor=None, facecolor='none', linewidth=1, **kwargs):
        if edgecolor is None:
            edgecolor = self.color

        (c, r) = (self.icol.start, self.irow.start)
        dc = self.icol.stop - self.icol.start
        dr = self.irow.stop - self.irow.start

        patch = Rectangle(
            (c, r), dc, dr,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            **kwargs)

        return patch

    def extract(self, f, key):
        return f[key][self.irow, self.icol, :]

    def df(self, f, key, melt=False):
        wn = f.attrs['wavenumber']
        assert len(f[key].shape) == 3
        assert f[key].shape[-1] == len(wn)

        foo = pd.DataFrame(
            self.extract(f, key).reshape(-1, len(wn)),
            columns=pd.Index(wn, name='wn')
        )

        if melt:
            return foo.melt(ignore_index=False).reset_index()

        return foo


def plot_sample(path, key='raw', filters=None, nrows=1, axes=None):
    if filters is None:
        filters = [
            (1000, 1200),
            (1200, 1300),
            (1350, 1500),
            (1500, 1600),
            (1600, 1700)
        ]


    ncols = len(filters) // nrows
    
    if axes is None:
        _, axes = plt.subplots(
            1, len(filters),
            figsize=(6.8*len(filters), 4.8*1),
            sharex=False, sharey=False)

    with h5py.File(path, 'r') as f:
        wn = f.attrs['wavenumber']

        for ax, (wn_start, wn_end) in zip(axes.ravel(), filters):
            start, end = np.argmax(wn > wn_start), np.argmax(wn > wn_end)
            ax.imshow(f[key][:, :, start:end].mean(2))
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title(f'({wn_start}, {wn_end})')

    return axes
