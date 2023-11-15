import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from irtoolkit.utils import Image


def main(outpath, image, config):
    fg = image.fgmask()
    signal = image.values[fg]

    if "pca_project" in config:
        pca = PCA(n_components=config["pca_project"]["n_components"])
        signal = pca.fit_transform(signal)

    assert "method" in config
    assert config["method"]["name"] == "kmeans-ablation"

    with h5py.File(outpath, "w") as f:
        for k in config["method"]["ks"]:
            labels = np.zeros(len(image.values), dtype=np.uint8)
            labels[fg] = KMeans(n_clusters=k, n_init="auto").fit_predict(signal) + 1
            f.create_dataset(f"kmeans-{k}", data=labels.reshape(image.shape[:2]))

    return


def inputs(args):
    outroot = Path("./data/chemical-images/clusters/")
    outroot.mkdir(exist_ok=True, parents=True)

    for path in map(Path, args.path):
        for cfg in map(Path, args.config):
            with open(cfg, "r") as f:
                config = yaml.safe_load(f)["cluster"]
                flavor = config.get("flavor", cfg.stem)

            with h5py.File(path, "r") as f:
                outpath = outroot / f"{path.stem}-{flavor}.h5"
                # if outpath.exists():
                #     continue
                image = Image(f["image"][:], f.attrs["wavenumber"])

            yield outpath, image, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="+", help="Apply to path (*s)", required=True)
    parser.add_argument("--config", nargs="+", help="Config (*s) to use", required=True)

    args = parser.parse_args()
    for outpath, image, config in inputs(args):
        main(outpath, image, config)
