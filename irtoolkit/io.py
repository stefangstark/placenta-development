from pathlib import Path

root = Path("./data/chemical-images/")


def qcroot(step):
    return root / "QC" / step


def path(sample, norm="raw"):
    if norm == "raw":
        outpath = root / "0-raw" / f"sample-{sample}-raw.h5"
    else:
        outpath = root / "1-normalized" / f"sample-{sample}-{norm}.h5"

    return outpath


def cluster(sample, norm, flavor):
    return root / "2-clustered" / f"clusters-{sample}-{norm}-{flavor}.h5"
