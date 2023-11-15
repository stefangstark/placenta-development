import yaml
import pandas as pd
from copy import deepcopy
import h5py
import argparse
from pathlib import Path
from irtoolkit import preprocess as pp, viz, qc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from irtoolkit.utils import Image


class QC:
    def __init__(self, regions, config):
        self.regions = regions
        self.steps = list()
        self.palette = {r.name: r.color for r in regions}
        self.config = config

        self.labels = (
            pd.concat([pd.Series(r.name, index=r.index) for r in self.regions])
            .sort_index()
            .rename_axis(index="pixel")
            .rename("region")
        )
        assert not self.labels.index.duplicated().any()

        self.sampled_labels = self.labels.groupby(self.labels).sample(
            500, random_state=1001287
        )

    def qc_fgmask(self, outpath, image, keep, parts):
        snr, pos, neg = parts
        index = self.labels.index

        df = pd.DataFrame(
            {
                "snr": snr[index],
                "pos": pos[index],
                "neg": neg[index],
                "keep": keep[index],
            },
            index=index,
        )
        df["region"] = self.labels

        _, axes = plt.subplots(2, 2, figsize=viz.scale_figsize(2, 2))
        ax = axes[0, 0]
        sns.histplot(
            x=df["pos"] - df["neg"],
            hue=df["region"],
            bins=25,
            palette=self.palette,
            ax=ax,
        )
        ax.axvline(config["mask_fg"]["threshold"], color="lightgrey")
        ax.set_title("signal to noise cutoff")

        ax = axes[0, 1]

        sns.scatterplot(
            data=df.groupby("region").sample(500).sample(frac=1),
            x="pos",
            y="neg",
            hue="region",
            s=5,
            palette=self.palette,
            ax=ax,
        )
        ax.set_title("positive vs negative signals")

        ax = axes[1, 0]
        signal = image.average_signal(1640, 1660).reshape(image.shape[:2])
        ax.imshow(signal)
        ax.set_title("slide (1640, 1660)")

        ax = axes[1, 1]
        ax.imshow(keep.reshape(image.shape[:2]).astype(np.int8), cmap="Greys")
        ax.set_title("foreground mask")

        plt.savefig(outpath, bbox_inches="tight")

        return

    def qc_pipeline_step(self, image, key):
        if self.regions is None:
            return

        df = pd.DataFrame(
            image.values[self.sampled_labels.index],
            columns=pd.Index(image.wn, name="wn"),
            index=pd.MultiIndex.from_frame(self.sampled_labels.reset_index()),
        )

        self.steps.append((key, df))

        return

    def finalize_pipeline(self, outpath):
        if self.regions is None:
            return

        palette = {box.name: box.color for box in self.regions}
        outpath.parent.mkdir(exist_ok=True, parents=True)

        ncols = len(self.steps)
        _, axes = viz.create_axes_grid(2 * ncols, ncols, sharex=False, sharey=False)
        for ax, (key, df) in zip(axes[0], self.steps):
            melt = df.dropna().melt(ignore_index=False).reset_index()
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

            ax.set_title(key)

        for ax, (_, df) in zip(axes[1], self.steps):
            sdf = df.dropna()
            pca = PCA(2)
            pcs = pd.DataFrame(
                pca.fit_transform(sdf), columns=["PCA 0", "PCA 1"], index=sdf.index
            ).reset_index()

            sns.scatterplot(
                data=pcs.sample(frac=1),  # shuffle dataset
                x="PCA 0",
                y="PCA 1",
                hue="region",
                palette=palette,
                legend=ax == axes[1, -1],
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


def main(path, config, qc_only=False):
    flavor = config["flavor"]
    pipeline = deepcopy(config["pipeline"])

    with h5py.File(path, "r") as f:
        sample = f.attrs["sample"]
        outpath = path.parent / f"norm-{flavor}-{sample}.h5"

        image = Image(f["image"][:], f.attrs["wavenumber"])

    qcdir = outpath.parent / "QC" / "1-normalize"
    qcdir.mkdir(exist_ok=True, parents=True)
    qcer = QC(qc.get_regions(sample), config)

    keep, parts = pp.signal_to_noise(image.values, image.wn, **config["mask_fg"])
    qcer.qc_fgmask(qcdir / f"{sample}-{flavor}-fgmask.png", image, keep, parts)

    image.values[~keep] = np.nan

    if qc_only:
        keep = qcer.sampled_labels.index

    qcer.qc_pipeline_step(image, "raw")

    for kwargs in pipeline:
        step = kwargs.pop("step")
        if step == "baseline_correction":
            image.values[keep] = pp.baseline_correction(
                image.values[keep], image.wn, **kwargs
            )

        elif step == "min_max_scale":
            image.values[keep] = pp.min_max_scale(image.values[keep], **kwargs)

        else:
            raise ValueError

        qcer.qc_pipeline_step(image, step)

    qcer.finalize_pipeline(qcdir / f"{sample}-{flavor}-pipeline.png")

    if qc_only:
        return

    with h5py.File(outpath, "w") as f:
        f.attrs["sample"] = sample
        f.attrs["wavenumber"] = image.wn
        f.create_dataset("image", data=image.values.reshape(image.shape))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="+", help="Apply to path (*s)", required=True)
    parser.add_argument("--config", nargs="+", help="Config (*s) to use", required=True)
    parser.add_argument(
        "--qc-only", action="store_true", help="Just run QC pipeline", dest="qconly"
    )

    args = parser.parse_args()
    for path in map(Path, args.path):
        for cfg in args.config:
            with open(cfg, "r") as f:
                config = yaml.safe_load(f)["normalize"]
            main(path, config, args.qconly)
