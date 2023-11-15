import numpy as np


class Image:
    def __init__(self, image, wn):
        self.shape = image.shape
        self.values = image.reshape(-1, len(wn))
        self.wn = wn

    def average_signal(self, start, stop):
        return self.values[:, wnslice(self.wn, start, stop)].mean(1)

    def fgmask(self):
        return np.isfinite(self.values).all(1)


def wnslice(wn, start, stop):
    return slice(np.argmax(wn > start), np.argmax(wn > stop))


def average_signal(f, key="image", irow=None, icol=None, start=1640, stop=1660):
    wn = f.attrs["wavenumber"][:]
    iwn = wnslice(wn, start, stop)
    signal = f[key][irow, icol, iwn].mean(2)

    return signal
