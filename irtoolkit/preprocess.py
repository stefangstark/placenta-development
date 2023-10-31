import cv2
import numpy as np

def extract_signal(f, key='raw', wn_start=1500, wn_end=1700):
    
    wn = f.attrs['wavenumber']
    start, end = np.argmax(wn > wn_start), np.argmax(wn > wn_end)
    signal = f[key][:, :, start:end].mean(2)

    return signal


def denoise(signal, threshold, kernel_size, strategy):

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = (signal > threshold).astype(np.uint8)

    if strategy == 'opening':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    elif strategy == 'closing':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    else:
        raise NotImplementedError


def min_max_scaling(signal, wn=None):
    shape = signal.shape
    if len(shape) == 3:
        signal = signal.reshape(-1, shape[2])

    assert signal.shape == 2

    shift = signal.min(1)[:, np.newaxis]
    scale = signal.max(1)[:, np.newaxis] - shift

    normed = (signal - shift) / scale

    if len(shape) == 3:
        return normed.reshape(shape)

    return normed


def amide_normalization(signal, wn=None, peak='I'):
    assert peak in {'I', 'II'}
    return
