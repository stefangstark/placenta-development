import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


class Animator:
    def __init__(
        self,
        projector,
        control,
        treated,
        start,
        end,
        ax,
        weights=None,
        title=None,
        xlabel="",
        ylabel="",
        **kwargs
    ):
        self.ax = ax
        self.title = title
        if projector == "pca":
            pca = PCA(n_components=2)
            self.base = pd.DataFrame(pca.fit_transform(np.vstack([control, treated])))
            self.projector = pca.transform

        else:
            self.projector = projector
            self.base = pd.DataFrame(self.projector(np.vstack([control, treated])))

        self.base["treatment"] = ["control"] * len(control) + ["treated"] * len(treated)

        self.start = start
        self.end = end

        self.kwargs = kwargs
        self.weights = [1] * len(self.base) + list(weights)

        self.xlim, self.ylim = None, None
        self.xlabel = xlabel
        self.ylabel = ylabel

    def preformat_ax(self):
        self.ax.clear()
        self.ax.spines[["right", "top"]].set_visible(False)

        if self.title is not None:
            self.ax.set_title(self.title)

        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)

        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

    def postformat_ax(self):
        if self.xlim is None:
            self.xlim = self.ax.get_xlim()

        if self.ylim is None:
            self.ylim = self.ax.get_ylim()

        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlabel(self.xlabel)

    def animate(self, lmda):
        self.preformat_ax()

        interpolate = (1 - lmda) * self.start + lmda * self.end
        shifted = pd.DataFrame(self.projector(interpolate))
        shifted["treatment"] = "predicted"

        data = pd.concat([self.base, shifted])

        sns.kdeplot(
            data,
            x=0,
            y=1,
            hue="treatment",
            weights=self.weights,
            **self.kwargs,
            ax=self.ax
        )

        self.postformat_ax()
        return
