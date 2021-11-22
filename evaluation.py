import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    houses = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
    truth = np.genfromtxt('resources/dataset_truth.csv',
                          delimiter=',', names=True, dtype=None,
                          encoding='UTF-8',
                          )
    truth = truth['Hogwarts_House']
    pred = np.genfromtxt('houses.csv', delimiter=',', names=True, dtype=None,
                         encoding='UTF-8')
    pred = pred['Hogwarts_House']
    print("Your score on test set: {}".format(accuracy_score(truth, pred)))
