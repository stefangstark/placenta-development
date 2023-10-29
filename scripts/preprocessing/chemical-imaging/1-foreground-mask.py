import sys
import h5py
from pathlib import Path
import yaml
from irtoolkit.preprocess import extract_signal, denoise
import argparse
from itertools import product


def run(path, config, name):
    with h5py.File(path, 'r+') as f:
        masks = f.require_group('foreground-masks')
        if name in masks:
            return

        signal = extract_signal(f, **config['extract-signal'])
        foreground_mask = denoise(signal, **config['denoise'])
        dset = masks.require_dataset(
                name,
                shape=foreground_mask.shape,
                dtype=foreground_mask.dtype
                )
        dset[:] = foreground_mask
        masks.attrs['config'] = yaml.dump(config)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', nargs='+',
        help='Paths to append mask',
        required=True)

    parser.add_argument(
        '--config', nargs='+',
        help='Config to use',
        required=True)

    args = parser.parse_args()
    for path, config in product(args.path, args.config):
        config = Path(config)
        name = config.stem
        print(path, name)
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)['make-foreground-mask']

        run(path, cfg, name)
