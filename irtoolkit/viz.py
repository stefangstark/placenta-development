import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from matplotlib.patches import Rectangle


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
