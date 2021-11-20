import random

import numpy as np
from matplotlib import pyplot as plt


def scatterPlot(courseX, courseY, dataset):
    legend = {
        'F-Test': '',
        'Gryffindor': 'brown',
        'Hufflepuff': 'black',
        'Ravenclaw': 'steelblue',
        'Slytherin': 'green'}
    for house, color in legend.items():
        if house == 'F-Test':
            continue
        data = dataset[dataset['Hogwarts_House'] == house][[courseX, courseY]]
        data = data[~np.isnan(data[courseX]) & ~np.isnan(data[courseY])]
        X = data[courseX]
        X = X[~np.isnan(X)]
        Y = data[courseY]
        Y = Y[~np.isnan(Y)]
        plt.scatter(X, Y, alpha=0.7, label=house, color=color)
        plt.xlabel(courseX)
        plt.ylabel(courseY)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    course = ['Arithmancy', 'Astronomy', 'Herbology',
              'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies',
              'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
              'Care_of_Magical_Creatures', 'Charms', 'Flying']
    dataset = np.genfromtxt('./resources/dataset_train.csv', delimiter=',',
                            names=True, filling_values=None, dtype=None,
                            encoding='UTF-8')
    courses = random.sample(course, 2)
    scatterPlot(courses[0], courses[1], dataset)
