from pathlib import Path

root = Path('./data/chemical-images/merged')


def path(sample, norm='raw'):
    if norm == 'raw':
        path = root/f'raw-{sample}.h5'
    else:
        path = root/f'norm-{norm}-{sample}.h5'

    return path
