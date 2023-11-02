import cv2
import numpy as np
from pybaselines import Baseline
from tqdm import tqdm

def extract_signal(f, key='raw', wn_start=1500, wn_end=1700):
    
    wn = f.attrs['wavenumber']
    start, end = np.argmax(wn > wn_start), np.argmax(wn > wn_end)
    signal = f[key][:, :, start:end].mean(2)

    return signal


def denoise(mask, kernel_size, strategy):
    if mask.dtype == bool:
        mask = mask.astype()

    kernel = np.ones((kernel_size, kernel_size), dtype=mask.dtype)

    if strategy == 'opening':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    elif strategy == 'closing':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    else:
        raise NotImplementedError


def min_max_scale(signal, wn=None):
    shape = signal.shape
    if len(shape) == 3:
        signal = signal.reshape(-1, shape[2])

    assert signal.ndim == 2

    shift = signal.min(1)[:, np.newaxis]
    scale = signal.max(1)[:, np.newaxis] - shift

    normed = (signal - shift) / scale

    if len(shape) == 3:
        return normed.reshape(shape)

    return normed


def rubberband_correct(signal, wn, flavor, *args, **kwargs):
    assert signal.shape[-1] == len(wn)
    fitter = Baseline(wn, check_finite=False)

    if flavor == 'modpoly':

        def fit(y, poly_order=1):
            return fitter.modpoly(y, poly_order=poly_order)[0]

    elif flavor == 'asls':

        def fit(y, lam=1e7, p=0.02):
            return fitter.asls(y, lam=lam, p=p)[0]

    elif flavor == 'mor':

        def fit(y, half_window=30):
            return fitter.mor(y, half_window=half_window)[0]

    elif flavor == 'snip':

        def fit(y, max_half_window=40, decreasing=True, smooth_half_window=3):
            return fitter.snip(y,
                max_half_window=max_half_window,
                decreasing=decreasing,
                smooth_half_window=smooth_half_window)[0]

    else:
        raise ValueError

    shape = signal.shape
    if len(shape) == 3:
        signal = signal.reshape(-1, len(wn))

    assert signal.ndim == 2
    rb = np.zeros_like(signal)
    for i, y in enumerate(tqdm(signal, desc='rubberband')):  # will be very slow, can vectorize
        rb[i] = y - fit(y, *args, **kwargs)

    if len(shape) == 3:
        return rb.reshape(shape)

    return rb
