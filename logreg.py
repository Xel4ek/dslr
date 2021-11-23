import numpy as np

from funcs import _mean, _std


class Scaler:
    def __init__(self, mean=np.array([]), std=np.array([])):
        self.mean = mean
        self.std = std

    def fit(self, x):
        for i in range(0, x.shape[1]):
            self.mean = np.append(self.mean, _mean(x[:, i]))
            self.std = np.append(self.std, _std(x[:, i]))

    def transform(self, x):
        return (x - self.mean) / self.std


def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g


class LogisticRegression:
    def __init__(self, eta=0.1, max_iter=50, lmb=0, initial_weight=None,
                 multi_class=None):
        self.eta = eta
        self.max_iter = max_iter
        self.lmb = lmb
        self._weights = initial_weight
        self._classes = multi_class
        self._errors = []
        self._cost = []

    def fit(self, x, y, sample_weight=None):
        self._classes = np.unique(y).tolist()
        new_x = np.insert(x, 0, 1, axis=1)
        m = new_x.shape[0]

        self._weights = sample_weight
        if not self._weights:
            self._weights = np.zeros(new_x.shape[1] * len(self._classes))
        self._weights = self._weights.reshape(len(self._classes), new_x.shape[1])

        y_vec = np.zeros((len(y), len(self._classes)))
        for i in range(0, len(y)):
            y_vec[i, self._classes.index(y[i])] = 1

        for _ in range(0, self.max_iter):
            predictions = self.net_input(new_x).T

            left = y_vec.T.dot(np.log(predictions))
            right = (1 - y_vec).T.dot(np.log(1 - predictions))

            r1 = (self.lmb / (2 * m)) * sum(sum(self._weights[:, 1:] ** 2))
            cost = (-1 / m) * sum(left + right) + r1
            self._cost.append(cost)
            self._errors.append(sum(y != self.predict(x)))

            r2 = (self.lmb / m) * self._weights[:, 1:]
            self._weights = self._weights - (
                    self.eta * (1 / m)
                    * (predictions - y_vec).T.dot(new_x)
                    + np.insert(r2, 0, 0, axis=1)
            )
        return self

    def net_input(self, x):
        return sigmoid(self._weights.dot(x.T))

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        predictions = self.net_input(x).T
        return [self._classes[x] for x in predictions.argmax(1)]

    def export_weights(self, sc, courses, filename='weights.csv'):
        with open(filename, 'w+') as f:
            for i in range(0, len(self._classes)):
                f.write(f'{self._classes[i]},')
            f.write('Mean,Std,Course\n')

            for j in range(0, self._weights.shape[1]):
                for i in range(0, self._weights.shape[0]):
                    f.write(f'{self._weights[i][j]},')
                f.write(
                    f'{sc.mean[j - 1] if j > 0 else ""},'
                    f'{sc.std[j - 1] if j > 0 else ""},'
                    f'{courses[j - 1] if j != 0 else ""}\n')

        return self

    def save_model_new(self, sc, features, filename='weights.csv'):
        with open(filename, 'w+') as f:
            f.write('Hogwarts_House, beta zero, ' + ', '.join(features) + '\n')

            for i, house in enumerate(self._classes):
                f.write(','.join([house, *[str(x) for x in self._weights[i]]]) + '\n')
            f.write('Mean,,' + ','.join([str(x) for x in sc.mean]) + '\n')
            f.write('Std,,' + ','.join([str(x) for x in sc.std]) + '\n')
        return self


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
    return x_train, x_test, y_train, y_test
