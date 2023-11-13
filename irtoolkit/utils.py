import numpy as np


class Image:
    def __init__(self, values, wn):
        if isinstance(wn, (int, np.integer)):
            if len(values.shape) == 3:
                assert values.shape[2] == 1
                values = values[:, :, 0]
            assert len(values.shape) == 2
        else:
            wn = list(wn)
            assert len(values.shape) == 3
            assert len(wn) == values.shape[2]

        self.values = values
        self.wn = wn
        self.shape = self.values.shape

    def __getitem__(self, arg):
        x, y, wn = arg

        if isinstance(wn, slice):
            assert wn.step is None
            assert isinstance(self.wn, list)
            start = self.wn.index(wn.start) if wn.start is not None else None
            stop = self.wn.index(wn.stop) if wn.stop is not None else None
            iwn = slice(start, stop, None)
            wn = self.wn[iwn]
            return self.values.__getitem__((x, y, iwn))

        elif isinstance(wn, int):
            iwn = self.wn.index(wn)
            wn = self.wn[iwn]
            return self.values.__getitem__((x, y, iwn))

    def __repr__(self):
        return self.values.__repr__()
