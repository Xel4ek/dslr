import numpy as np

from funcs import _mean, _std


class Scaler:
    def __init__(self, mean=np.array([]), std=np.array([])):
        self._mean = mean
        self._std = std

    def fit(self, x):
        for i in range(0, x.shape[1]):
            self._mean = np.append(self._mean, _mean(x[:, i]))
            self._std = np.append(self._std, _std(x[:, i]))

    def transform(self, x):
        return ((x - self._mean) / self._std)


class LogisticRegression:
    def __init__(self, eta=0.1, max_iter=50, Lambda=0, initial_weight=None,
                 multi_class=None):
        self.eta = eta
        self.max_iter = max_iter
        self.Lambda = Lambda
        self._w = initial_weight
        self._K = multi_class
        self._errors = []
        self._cost = []

    def fit(self, x, y, sample_weight=None):
        self._K = np.unique(y).tolist()
        newX = np.insert(x, 0, 1, axis=1)
        m = newX.shape[0]

        self._w = sample_weight
        if not self._w:
            self._w = np.zeros(newX.shape[1] * len(self._K))
        self._w = self._w.reshape(len(self._K), newX.shape[1])

        yVec = np.zeros((len(y), len(self._K)))
        for i in range(0, len(y)):
            yVec[i, self._K.index(y[i])] = 1

        for _ in range(0, self.max_iter):
            predictions = self.net_input(newX).T

            lhs = yVec.T.dot(np.log(predictions))
            rhs = (1 - yVec).T.dot(np.log(1 - predictions))

            r1 = (self.Lambda / (2 * m)) * sum(sum(self._w[:, 1:] ** 2))
            cost = (-1 / m) * sum(lhs + rhs) + r1
            self._cost.append(cost)
            self._errors.append(sum(y != self.predict(x)))

            r2 = (self.Lambda / m) * self._w[:, 1:]
            self._w = self._w - (
                    self.eta * (1 / m) * (predictions - yVec).T.dot(
                newX) + np.insert(r2, 0, 0, axis=1))
        return self

    def net_input(self, X):
        return self.sigmoid(self._w.dot(X.T))

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        predictions = self.net_input(X).T
        return [self._K[x] for x in predictions.argmax(1)]

    def save_model(self, sc, features, filename='weights.csv'):
        with open(filename, 'w+') as f:
            f.write('Hogwarts_House, beta zero, ' + ', '.join(features) + '\n')

            for i, house in enumerate(self._K):
                f.write(','.join([house, *[str(x) for x in self._w[i]]]) + '\n')
            f.write('Mean,,' + ','.join([str(x) for x in sc._mean]) + '\n')
            f.write('Std,,' + ','.join([str(x) for x in sc._std]) + '\n')
        return self

    def sigmoid(self, z):
        g = 1.0 / (1.0 + np.exp(-z))
        return g