import re
import scipy as sp
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import h5py


@dataclass
class Patch:
    path: str
    slide: int
    region: str
    col: int
    row: int
    dx: int
    dy: int

    def load(self, idxs=None):
        lut = sp.io.loadmat(self.path, variable_names=['dX', 'dY', 'r'])
        r = lut['r']
        dx = int(lut['dX'].item())
        dy = int(lut['dY'].item())

        r = r.reshape((dx, dy, r.shape[1]))
        if idxs is not None:
            return r[:, :, idxs].copy()

        return r


def load_wn(patches):
    'Load wn (col names) & check that it is similar across all patches'
    def load_single_wn(patch):
        wn = sp.io.loadmat(patch.path, variable_names='wn')['wn'].ravel()
        return wn

    wn = load_single_wn(patches[0])
    for patch in patches[1:]:
        assert np.array_equal(wn, load_single_wn(patch))
    return wn


def map_loc_values_to_grid(paths):
    cols = set()
    rows = set()
    for path in paths:
        slide, region = re.search(r'Slide(\d)/Region(\w)', path).groups()
        py, px = re.search(r'\[(-?\d+)_(-?\d+)\]', path).groups()
        cols.add(int(px))
        rows.add(int(py))

    return sorted(rows), sorted(cols, reverse=True)


def load_patches(glob):
    paths = list(map(str, glob))
    rows, cols = map_loc_values_to_grid(paths)

    def iterate(paths):

        for path in paths:
            slide, region = re.search(r'Slide(\d)/Region(\w)', path).groups()
            py, px = re.search(r'\[(-?\d+)_(-?\d+)\]', path).groups()
            col, row = cols.index(int(px)), rows.index(int(py))

            lut = sp.io.loadmat(path, variable_names=['dX', 'dY'])
            dx = int(lut['dX'].item())
            dy = int(lut['dY'].item())

            patch = Patch(
                path=path,
                slide=slide,
                region=region,
                dx=dx, dy=dy,
                col=col, row=row
            )

            yield patch

    patches = {(p.row, p.col): p for p in iterate(paths)}
    assert len(set(p.dx for p in patches.values())) == 1
    assert len(set(p.dy for p in patches.values())) == 1

    for key in product(range(len(rows)), range(len(cols))):
        assert key in patches, f'Patch {key} missing'

    return cols, rows, patches


if __name__ == '__main__':
    inroot = Path('data/chemical-images/0-upload-daylight-solutions')
    outroot = Path('data/chemical-images/1-merged-regions')

    cols, rows, patches = load_patches(inroot.glob('Slide1/RegionJ*'))
    grid = np.zeros((len(rows)*480, len(cols)*480, 2))
    for patch in patches.values():
        dx, dy = patch.dx, patch.dy
        col, row = patch.col, patch.row
        grid[dx*row: dx*(row + 1), dy*col: dy*(col + 1)] = patch.load([100, 200])
    plt.imshow(grid[:, :, 1])
    plt.show()

    with h5py.File(outroot/'1J.hdf5', 'w') as f:
        grid = f.create_dataset('grid', (len(rows)*480, len(cols)*480, 2))
        for patch in patches.values():
            dx, dy = patch.dx, patch.dy
            col, row = patch.col, patch.row
            grid[dx*row: dx*(row + 1), dy*col: dy*(col + 1)] = patch.load([100, 200])

    with h5py.File(outroot/'1J.hdf5', 'r') as f:
        img = f['grid'][:]

    plt.imshow(img[:, :, 1])
    plt.show()
