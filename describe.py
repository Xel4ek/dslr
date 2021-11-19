import argparse
import csv

import numpy as np

from funcs import _count, _mean, _std, _min, _percentile, _max


def describe(filename):
  colWidth = 14
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
  features = ' '.join([f'{x:>{colWidth}}' for x in dataset.dtype.names])
  print(f'{"":7}\t' + features)
  for k, v in params.items():
    print(f'{k:<7}', end='\t', )
    dataStr = [f'{v(dataset[name]):>{max(len(name), colWidth)}.6f}' for name in dataset.dtype.names]
    print(*dataStr, sep=' ')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help="input dataset")
  args = parser.parse_args()

  describe(args.dataset)
