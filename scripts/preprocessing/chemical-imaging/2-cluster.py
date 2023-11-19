import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from irtoolkit import io, viz
from irtoolkit.utils import Image


def qc(qcpath, clusterpath, image):
    with h5py.File(clusterpath) as f:
        clusters = {k: f[k][:] for k in f.keys()}
        source = f.attrs["source"]

    with h5py.File(source, "r") as f:
        image = Image(f["image"][:], f.attrs["wavenumber"])

    _, axes = viz.create_axes_grid(len(clusters) * 2, len(clusters))

    axiter = zip(axes[::2].ravel(), axes[1::2].ravel())
    tab10 = plt.get_cmap("tab10")

    for (aximage, axline), labels in zip(axiter, clusters.values()):
        k = labels.max()
        cmap = ListedColormap(["#eeeeee"] + [tab10(i) for i in range(k)])
        aximage.imshow(labels, cmap=cmap)
        aximage.set_title(f"K={k}")

        labels = pd.Series(labels.ravel())
        samples = (
            labels.groupby(labels)
            .sample(500)
            .rename("label")
            .rename_axis(index="pixel")
            .reset_index()
        )
        index = pd.MultiIndex.from_frame(samples)

        df = pd.DataFrame(
            image.values[samples["pixel"]],
            columns=pd.Index(image.wn, name="wn"),
            index=index,
        )
        melt = df.dropna().melt(ignore_index=False).reset_index()
        sns.lineplot(
            melt,
            x="wn",
            y="value",
            hue="label",
            errorbar="pi",
            palette=dict(enumerate(cmap.colors)),
            ax=axline,
        )

    plt.savefig(qcpath, bbox_inches="tight")


def feature_extraction(signal, style, **kwargs):
    if style == "pca":
        pca = PCA(**kwargs)
        signal = pca.fit_transform(signal)
    else:
        raise ValueError

    return signal


def cluster(outpath, image, config):
    fg = image.fgmask()
    signal = feature_extraction(image.values[fg], **config["feature_extraction"])

    assert "method" in config
    assert config["method"]["style"] == "kmeans-ablation"
    ks = config["method"]["ks"]

    with h5py.File(outpath, "w") as f:
        f.attrs["source"] = str(image.path)
        for k in ks:
            labels = np.zeros(len(image.values), dtype=np.uint8)
            labels[fg] = KMeans(n_clusters=k, n_init="auto").fit_predict(signal) + 1
            f.create_dataset(f"kmeans-{k}", data=labels.reshape(image.shape[:2]))


def main(outpath, qcpath, image, config, qconly=False):
    if not qconly:
        cluster(outpath, image, config)
    assert outpath.exists()

    qc(qcpath, outpath, image)

    return


def inputs(args):
    qcroot = io.qcroot("2-cluster")
    qcroot.mkdir(exist_ok=True, parents=True)

    for inpath in map(Path, args.path):
        for cfg in map(Path, args.config):
            with open(cfg, "r") as f:
                config = yaml.safe_load(f)["cluster"]
                flavor = config.get("flavor", cfg.stem)

            with h5py.File(inpath, "r") as f:
                outpath = io.cluster(f.attrs["sample"], f.attrs["norm"], flavor)
                outpath.parent.mkdir(exist_ok=True, parents=True)

                qcpath = qcroot / f"{outpath.stem}.png"
                if outpath.exists() and qcpath.exists():
                    continue
                image = Image(f["image"][:], f.attrs["wavenumber"])
                image.path = inpath

            yield outpath, qcpath, image, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="+", help="Apply to path (*s)", required=True)
    parser.add_argument("--config", nargs="+", help="Config (*s) to use", required=True)
    parser.add_argument("--qc-only", action="store_true", dest="qc_only")

    args = parser.parse_args()
    for clusterpath, qcpath, image, config in inputs(args):
        main(clusterpath, qcpath, image, config, args.qc_only)
