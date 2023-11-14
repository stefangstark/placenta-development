import yaml
import pandas as pd
from copy import deepcopy
import h5py
import argparse
from pathlib import Path
from irtoolkit import preprocess as pp, viz
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class QC:
    def __init__(self, regions):
        self.regions = regions
        self.steps = list()

    def step(self, signal, wn, key):
        if self.regions is None:
            return

        def iterate():
            for box in self.regions:
                df = (
                    pd.DataFrame(box.mask(signal, flatten=True), columns=wn)
                    .sample(1000, random_state=1001231)
                    .rename_axis(index="pixel")
                    .reset_index()
                )
                df["region"] = box.name
                yield df

        df = pd.concat(iterate())
        self.steps.append((key, df))

        return

    def finalize(self, outpath):
        if self.regions is None:
            return

        palette = {box.name: box.color for box in self.regions}
        outpath.parent.mkdir(exist_ok=True, parents=True)

        ncols = len(self.steps)
        _, axes = viz.create_axes_grid(2 * ncols, ncols, sharex=False, sharey=False)
        for ax, (_, df) in zip(axes[0], self.steps):
            melt = (
                df.set_index(["pixel", "region"])
                .melt(ignore_index=False, var_name="wn")
                .reset_index()
            )
            melt["wn"] = melt["wn"].astype(int)
            sns.lineplot(
                data=melt,
                x="wn",
                y="value",
                hue="region",
                errorbar="pi",
                legend=False,
                ax=ax,
            )

        for ax, (_, df) in zip(axes[1], self.steps):
            pca = PCA(2)
            pcs = pd.DataFrame(pca.fit_transform(df.drop(columns=["pixel", "region"])))
            pcs["region"] = df["region"].values

            sns.scatterplot(
                data=pcs.sample(frac=1),  # shuffle dataset
                x=0,
                y=1,
                hue="region",
                palette=palette,
                alpha=0.25,
                s=5,
                legend=False,
                ax=ax,
            )

        axes[0, -1].legend(
            handles=viz.legend_from_lut(palette),
            bbox_to_anchor=(1, 1),
            loc="upper left",
            title="Region",
        )

        plt.savefig(outpath, bbox_inches="tight")

        return


class Image:
    def __init__(self, image, wn):
        self.shape = image.shape
        self.values = image.reshape(-1, len(wn))
        self.wn = wn

    def average(self, start, stop):
        istart, istop = np.argmax(self.wn > start), np.argmax(self.wn > stop)
        return self.values[:, istart:istop].mean(1)


def main(path, config):
    flavor = config["flavor"]
    pipeline = deepcopy(config["pipeline"])

    with h5py.File(path, "r") as f:
        sample = f.attrs["sample"]
        outpath = path.parent / f"norm-{flavor}-{sample}.h5"

        image = Image(f["image"][:], f.attrs["wavenumber"])

    qcdir = outpath.parent / 'QC' / '1-normalize'
    qcdir.mkdir(exist_ok=True, parents=True)

    keep, _ = pp.signal_to_noise(image.values, image.wn, **config["mask_fg"])
    image.values[~keep] = np.nan
    
    for kwargs in pipeline:
        name = kwargs.pop("step")
        if name == "baseline_correction":
            image.values[keep] = pp.baseline_correction(image.values[keep], image.wn, **kwargs)

        elif name == "min_max_scale":
            image.values[keep] = pp.min_max_scale(image.values[keep], **kwargs)

        else:
            raise ValueError

    with h5py.File(outpath, "w") as f:
        f.attrs["sample"] = sample
        f.attrs["wavenumber"] = image.wn
        f.create_dataset("image", data=image.values.reshape(image.shape))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="+", help="Apply to path (*s)", required=True)
    parser.add_argument("--config", nargs="+", help="Config (*s) to use", required=True)

    args = parser.parse_args()
    for path in map(Path, args.path):
        for cfg in args.config:
            with open(cfg, "r") as f:
                config = yaml.safe_load(f)["normalize"]
            main(path, config)
