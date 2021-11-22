import argparse

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score

from logreg import Scaler, LogisticRegression


def train_test_split(x, y, test_size, random_state):
    # Split random train and test subsets

    if len(x) != len(y):
        raise Exception(f'Size of arrays should be equal: {len(x)} != {len(y)}')

    if random_state:
        np.random.seed(random_state)

    perm = np.random.permutation(len(x))

    offset = int(len(x) * test_size)

    x_train = x[perm][offset:]
    x_test = x[perm][:offset]

    y_train = y[perm][offset:]
    y_test = y[perm][:offset]
    return (x_train, x_test, y_train, y_test)

def prepare_dataset(courses, dataset: DataFrame):
    dataset.dropna(subset=courses, inplace=True)
    x = np.array(dataset[courses], dtype=float)
    y = np.array(dataset['Hogwarts House'])
    return (x, y)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Train dataset")
    parser.add_argument("-courses", type=str, help="Selected courses")
    args = parser.parse_args()

    dataset = pd.read_csv('resources/dataset_train.csv')

    if args.courses:
        with open(args.courses) as f:
            courses = f.read().split('\n')
    else:
        courses = [
            'Herbology',
            'Defense Against the Dark Arts',
            'Muggle Studies',
            'Ancient Runes',
            'Divination',
            'Charms'
        ]
    return (courses, dataset)


if __name__ == '__main__':
    courses, dataset = parse_args()
    x, y = prepare_dataset(courses, dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=4)
    sc = Scaler()
    sc.fit(x_train)

    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    lr = LogisticRegression(eta=0.01, max_iter=50, Lambda=10)
    lr.fit(x_train_std, y_train)

    y_pred = lr.predict(x_test_std)
    print(f'Wrong predictions: {sum(y_test != y_pred)}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

    lr.save_model(sc, courses)
