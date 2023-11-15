from pathlib import Path

root = Path("./data/chemical-images/")


def qcroot(step):
    return root / "QC" / step


def path(sample, norm="raw"):
    return root / "samples" / f"sample-{sample}-{norm}.h5"
