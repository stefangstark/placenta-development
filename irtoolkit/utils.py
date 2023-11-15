import numpy as np


class Image:
    def __init__(self, image, wn):
        self.shape = image.shape
        self.values = image.reshape(-1, len(wn))
        self.wn = wn

    def average_signal(self, start, stop):
        return self.values[:, wnrange(self.wn, start, stop)].mean(1)


def wnrange(wn, start, stop):
    return slice(np.argmax(wn > start), np.argmax(wn > stop))


def average_signal(f, key="image", irow=None, icol=None, start=1640, stop=1660):
    wn = f.attrs["wavenumber"][:]
    iwn = wnrange(wn, start, stop)
    signal = f[key][irow, icol, iwn].mean(2)

    return signal
