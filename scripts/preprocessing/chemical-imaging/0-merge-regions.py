import re
import scipy as sp
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import h5py
import argparse


@dataclass
class Patch:
    path: str
    slide: int
    region: str
    col: int
    row: int

    def load(self, idxs=None):
        lut = sp.io.loadmat(self.path, variable_names=['dX', 'dY', 'r'])
        r = lut['r']
        dx = int(lut['dX'].item())
        dy = int(lut['dY'].item())

        r = r.reshape((dx, dy, r.shape[1]))
        if idxs is not None:
            return r[:, :, idxs].copy()

        return r


def map_loc_values_to_grid(paths):
    cols = set()
    rows = set()
    for path in paths:
        slide, region = re.search(r'Slide(\d)/Region(\w)', path).groups()
        py, px = re.search(r'\[(-?\d+)_(-?\d+)\]', path).groups()
        cols.add(int(px))
        rows.add(int(py))

    return sorted(rows), sorted(cols, reverse=True)


def check_patches_assumptions(patches):
    assert len(patches) > 0
    nrows = max(p.row for p in patches) + 1
    ncols = max(p.col for p in patches) + 1

    all_grid_locs = set(product(range(nrows), range(ncols)))
    for patch in patches:
        all_grid_locs.remove((patch.row, patch.col))
    assert len(all_grid_locs) == 0


def load_patches(glob):
    paths = list(map(str, glob))
    rows, cols = map_loc_values_to_grid(paths)

    def iterate(paths):

        for path in paths:
            slide, region = re.search(r'Slide(\d)/Region(\w)', path).groups()
            py, px = re.search(r'\[(-?\d+)_(-?\d+)\]', path).groups()
            col, row = cols.index(int(px)), rows.index(int(py))

            patch = Patch(
                path=path,
                slide=slide,
                region=region,
                col=col, row=row
            )

            yield patch

    patches = list(iterate(paths))
    check_patches_assumptions(patches)

    return patches


def load_patch_size(patches):
    'Aumming patches are squares, get their length'

    dx = load_key(patches, 'dX')
    dy = load_key(patches, 'dY')
    assert dx == dy
    return dx


def load_wn(patches):
    'Load wn (col names) & check that it is similar across all patches'
    def load_single_wn(patch):
        wn = sp.io.loadmat(patch.path, variable_names='wn')['wn'].ravel()
        return wn

    wn = load_single_wn(patches[0])
    for patch in patches[1:]:
        assert np.array_equal(wn, load_single_wn(patch))
    return wn


def load_key(patches, key):
    values = set()
    values.update([
        sp.io.loadmat(patch.path, variable_names=key)[key].item()
        for patch
        in patches
        ])
    assert len(values) == 1
    return values.pop()


def stitch_regions(outpath, glob):
    patches = load_patches(glob)
    nrows = max(p.row for p in patches) + 1
    ncols = max(p.col for p in patches) + 1
    patch_size = load_patch_size(patches)
    wn = load_wn(patches)

    shape = (nrows*patch_size, ncols*patch_size, len(wn))

    with h5py.File(outpath, 'w') as f:
        f.attrs['wavenumber'] = wn
        f.attrs['model'] = load_key(patches, 'model')
        grid = f.create_dataset('raw', shape)
        for patch in patches:
            col, row = patch.col, patch.row
            irow = slice(patch_size*row, patch_size*(row + 1))
            icol = slice(patch_size*col, patch_size*(col + 1))
            grid[irow, icol] = patch.load()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('slide', type=int)
    parser.add_argument('region')
    args = parser.parse_args()
    outdir = Path('./data/chemical-images/1-merged-regions/')
    outdir.parent.mkdir(exist_ok=True, parents=True)
    outpath = outdir/f'sample-slide-{args.slide}-region-{args.region}.h5ad'

    glob = (
            Path('./data/chemical-images/0-upload-daylight-solutions')
            .glob(f'Slide{args.slide}/Region{args.region}*')
            )

    stitch_regions(outpath, glob)
    with h5py.File(outpath, 'r') as f:
        wn = f.attrs['wavenumber'].tolist().index(1150)
        plt.imshow(f['raw'][:, :, wn])
        plt.savefig(outpath.parent/f'qc-{outpath.stem}.png')
