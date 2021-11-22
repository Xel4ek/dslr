import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def scatter_plot(ax: Axes, courseX, courseY, dataset):
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
        ax.scatter(X, Y, alpha=0.5, label=house, color=color, s=1)


if __name__ == '__main__':
    course = ['Arithmancy', 'Astronomy', 'Herbology',
              'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies',
              'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
              'Care_of_Magical_Creatures', 'Charms', 'Flying']
    dataset = np.genfromtxt('./resources/dataset_train.csv', delimiter=',',
                            names=True, filling_values=None, dtype=None,
                            encoding='UTF-8')
    courses = random.sample(course, 2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(course[0])
    ax.set_ylabel(course[1])
    scatter_plot(ax, courses[0], courses[1], dataset)
    plt.legend()
    plt.show()