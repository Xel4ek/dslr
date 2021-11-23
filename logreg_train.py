import argparse

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score

from logreg import Scaler, LogisticRegression, train_test_split


def prepare_dataset(courses, dataset: DataFrame):
    dataset.dropna(subset=courses, inplace=True)
    x = np.array(dataset[courses], dtype=float)
    y = np.array(dataset['Hogwarts House'])
    return x, y


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
            'Defense Against the Dark Arts',
            'Charms',
            'Herbology',
            # 'Divination',
            'Muggle Studies',
        ]
    return courses, dataset


if __name__ == '__main__':
    courses, dataset = parse_args()
    x, y = prepare_dataset(courses, dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=4)
    sc = Scaler()
    sc.fit(x_train)

    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    lr = LogisticRegression(eta=0.01, max_iter=50, lmb=10)
    lr.fit(x_train_std, y_train)

    y_pred = lr.predict(x_test_std)
    print(f'Wrong predictions: {sum(y_test != y_pred)}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

    lr.export_weights(sc, courses)
