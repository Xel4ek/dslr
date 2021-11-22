import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
   # Load the truths
   houses = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
   # convfunc = lambda x: int(houses[x])
   tr = np.genfromtxt('resources/dataset_truth.csv',
                      delimiter=',', names=True, dtype=None, encoding='UTF-8',
                      # converters={'Hogwarts_House': convfunc}
                      )
   pr = np.genfromtxt('houses.csv', delimiter=',', names=True, dtype=None, encoding='UTF-8')
   # truths = pd.read_csv('resources/dataset_truth.csv', sep=',', index_col=0)
   tr = tr['Hogwarts_House']
   pr = pr['Hogwarts_House']
   # Load predictions
   # predictions = pd.read_csv('houses.csv', sep=',', index_col=0)
   # Replace names by numerical value {0, 1, 2, 3} and convert to array
   # y_true = truths.replace(houses).as_matrix()
   # y_pred = predictions.replace(houses).as_matrix()
   # Print the score using accuracy_score
   print("Your score on test set: {}".format(accuracy_score(tr, pr)))