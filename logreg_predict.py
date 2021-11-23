import argparse

import numpy as np
import pandas as pd

from logreg import LogisticRegression, Scaler


def method_name():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="input dataset")
    parser.add_argument("weights", type=str, help="input weights")
    args = parser.parse_args()
    return args.dataset, args.weights


if __name__ == '__main__':
    dataset, weights = method_name()

    df = pd.read_csv(dataset)
    df = df.fillna(method='ffill')
    courses = [
      'Defense Against the Dark Arts',
      'Charms',
      'Herbology',
      # 'Divination',
      'Muggle Studies',
    ]
    x = np.array(df[courses].values, dtype=float)

    df = pd.read_csv(weights)
    k = list(df)[:4]
    mean = df.values[1:, 4]
    std = df.values[1:, 5]
    weights = df.values[:, :4].T

    sc = Scaler(mean, std)
    x_std = sc.transform(x)

    lr = LogisticRegression(initial_weight=weights, multi_class=k)

    y_pred = lr.predict(x_std)

    f = open("houses.csv", 'w+')
    f.write('Index,Hogwarts House\n')
    for i in range(0, len(y_pred)):
        f.write(f'{i},{y_pred[i]}\n')
