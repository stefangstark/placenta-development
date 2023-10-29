import cv2
import numpy as np

def extract_signal(f, key='raw', wn_start=1500, wn_end=1700):
    
    wn = f.attrs['wavenumber']
    start, end = np.argmax(wn > wn_start), np.argmax(wn > wn_end)
    signal = f[key][:, :, start:end].mean(2)

    return signal


def denoise(signal, threshold, kernel_size, strategy):
    if strategy == 'opening':
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = (signal > threshold).astype(np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    else:
        raise NotImplementedError
