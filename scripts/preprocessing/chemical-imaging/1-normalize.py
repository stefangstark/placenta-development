import yaml
import pandas as pd
import h5py
import argparse
from pathlib import Path
from irtoolkit import preprocess as pp, qc, viz
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class QC:
    def __init__(self, regions):
        self.regions = regions
        self.steps = list()

    def step(self, signal, wn, key):
        if self.regions is None:
            return

        def iterate():
            for box in self.regions:
                df = (
                    pd.DataFrame(box.mask(signal, flatten=True), columns=wn)
                    .sample(1000, random_state=1001231)
                    .rename_axis(index='pixel')
                    .reset_index()
                    )
                df['region'] = box.name
                yield df

        df = pd.concat(iterate())
        self.steps.append((key, df))

        return

    def finalize(self, outpath):
        if self.regions is None:
            return

        palette = {box.name: box.color for box in self.regions}
        outpath.parent.mkdir(exist_ok=True, parents=True)

        ncols = len(self.steps)
        fig, axes = viz.create_axes_grid(2*ncols, ncols, sharex=False, sharey=False)
        for ax, (step, df) in zip(axes[0], self.steps):
            melt = (
                df
                .set_index(['pixel', 'region'])
                .melt(ignore_index=False, var_name='wn')
                .reset_index()
            )
            melt['wn'] = melt['wn'].astype(int)
            sns.lineplot(
                    data=melt,
                    x='wn', y='value',
                    hue='region',
                    errorbar='pi',
                    legend=False,
                    ax=ax)
            
        for ax, (step, df) in zip(axes[1], self.steps):

            pca = PCA(2)
            pcs = pd.DataFrame(
                pca.fit_transform(df.drop(columns=['pixel', 'region']))
            )
            pcs['region'] = df['region'].values

            sns.scatterplot(
                data=pcs.sample(frac=1),  # shuffle dataset
                x=0, y=1,
                hue='region',
                palette=palette,
                alpha=0.25, s=5,
                legend=False,
                ax=ax
            )

        axes[0, -1].legend(
            handles=viz.legend_from_lut(palette),
            bbox_to_anchor=(1, 1),
            loc='upper left',
            title='Region')

        plt.savefig(outpath, bbox_inches='tight')

        return 

def main(path, flavor, pipeline):
    with h5py.File(path, 'r') as f:

        sample = f.attrs['sample']
        outpath = path.parent/f'norm-{flavor}-{sample}.h5'
        if outpath.exists():
            return

        wn = list(f.attrs['wavenumber'][:])
        signal = f['image'][:]


    qcer = QC(qc.get_regions(sample))
    
    qcer.step(signal, wn, f'input')
    for kwargs in pipeline:
        name = kwargs.pop('step')
        if name == 'rubberband-correct':
            step = pp.rubberband_correct

        elif name == 'min-max-scale':
            step = pp.min_max_scale

        else:
            raise ValueError

        signal = step(signal, wn, **kwargs)
        # for box in qcer.regions:
        #     signal[box.irow, box.icol] = step(signal[box.irow, box.icol], wn, **kwargs)
        qcer.step(signal, wn, name)

    qcer.finalize(path.parent/'QC'/'1-normalize'/f'{sample}.png')

    with h5py.File(outpath, 'w') as f:
        f.attrs['sample'] = sample
        f.attrs['wavenumber'] = wn
        f.create_dataset('image', data=signal)
        
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', nargs='+',
        help='Apply to path (*s)',
        required=True)

    parser.add_argument(
        '--config', nargs='+',
        help='Config (*s) to use',
        required=True)

    args = parser.parse_args()
    for path in map(Path, args.path):
        for cfg in args.config:
            with open(cfg, 'r') as f:
                config = yaml.safe_load(f)['normalize'] 
            main(path, **config)
