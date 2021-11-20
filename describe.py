import argparse

import numpy as np

from funcs import _count, _mean, _std, _min, _percentile, _max


def describe(filename):
    col_width = 14
    dataset = np.genfromtxt(filename, delimiter=',', names=True)
    params = {
        'Count': _count,
        'Mean': _mean,
        'Std': _std,
        'Min': _min,
        '25%': lambda data: _percentile(data, 25),
        '50%': lambda data: _percentile(data, 50),
        '75%': lambda data: _percentile(data, 75),
        'Max': _max,
    }
    features = ' '.join([f'{x:>{col_width}}' for x in dataset.dtype.names])
    print(f'{"":7}\t' + features)
    for k, v in params.items():
        print(f'{k:<7}', end='\t', )
        data_str = [f'{v(dataset[name]):>{max(len(name), col_width)}.6f}' for name in dataset.dtype.names]
        print(*data_str, sep=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="input dataset")
    args = parser.parse_args()

    describe(args.dataset)
