import cv2
import numpy as np
from pybaselines import Baseline
from tqdm import tqdm


def denoise(mask, kernel_size, strategy):
    if mask.dtype == bool:
        mask = mask.astype()

    kernel = np.ones((kernel_size, kernel_size), dtype=mask.dtype)

    if strategy == "opening":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    elif strategy == "closing":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    raise NotImplementedError


def signal_to_noise(values, wn, positive_range, negative_range, threshold):
    def average(start, stop):
        istart, istop = np.argmax(wn > start), np.argmax(wn > stop)
        return values[:, istart:istop].mean(1)

    pos = average(*positive_range)
    neg = average(*negative_range)
    snr = pos - neg
    keep = snr > threshold

    return keep, (snr, pos, neg)


def min_max_scale(values):
    assert values.ndim == 2

    shift = values.min(1)[:, np.newaxis]
    scale = values.max(1)[:, np.newaxis] - shift

    values = (values - shift) / scale

    return values


def baseline_correction(values, wn, flavor, *args, **kwargs):
    fitter = Baseline(wn, check_finite=False)

    if flavor == "modpoly":

        def fit(y, poly_order=1):
            return fitter.modpoly(y, poly_order=poly_order)[0]

    elif flavor == "asls":

        def fit(y, lam=1e7, p=0.02):
            return fitter.asls(y, lam=lam, p=p)[0]

    elif flavor == "mor":

        def fit(y, half_window=30):
            return fitter.mor(y, half_window=half_window)[0]

    elif flavor == "snip":

        def fit(y, max_half_window=40, decreasing=True, smooth_half_window=3):
            return fitter.snip(
                y,
                max_half_window=max_half_window,
                decreasing=decreasing,
                smooth_half_window=smooth_half_window,
            )[0]

    else:
        raise ValueError

    # HACK: computes bl individually over for loop, very slow
    corrected = np.zeros_like(values)
    for i, y in enumerate(tqdm(values, desc="baseline")):
        corrected[i] = y - fit(y, *args, **kwargs)

    return corrected
