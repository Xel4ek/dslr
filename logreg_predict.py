
from sklearn.metrics import accuracy_score

import argparse

import pandas as pd
import numpy as np

from logreg import Scaler, LogisticRegression

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help="input dataset")
  parser.add_argument("weights", type=str, help="input weights")
  args = parser.parse_args()

  df = pd.read_csv(args.dataset, index_col='Hogwarts House')
  df = df.fillna(method='ffill')

  df_weights = pd.read_csv(args.weights, index_col='Hogwarts_House')
  K = list(df_weights.index[:4])
  mean = df_weights.loc['Mean'].dropna().values
  std = df_weights.loc['Std'].dropna().values
  weights = df_weights.loc[K].dropna().values
  courses = list(df_weights.columns[2:])

  X = np.array(df[courses].values, dtype=float)
  sc = Scaler(mean, std)
  X_std = sc.transform(X)

  lr = LogisticRegression(initial_weight=weights,multi_class=K)

  y_pred = lr.predict(X_std)

  f = open("houses.csv", 'w+')
  f.write('Index,Hogwarts House\n')
  for i in range(0, len(y_pred)):
    f.write(f'{i},{y_pred[i]}\n')
