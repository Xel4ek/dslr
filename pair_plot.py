import numpy as np
from matplotlib import pyplot as plt

from histogram import histogram
from scatter_plot import scatter_plot


if __name__ == '__main__':
    legend = {
        'Gryffindor': 'brown',
        'Hufflepuff': 'black',
        'Ravenclaw': 'steelblue',
        'Slytherin': 'green'
    }
    course = ['Arithmancy', 'Astronomy', 'Herbology',
              'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies',
              'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
              'Care_of_Magical_Creatures', 'Charms', 'Flying']
    dataset = np.genfromtxt('./resources/dataset_train.csv', delimiter=',',
                            names=True, filling_values=None, dtype=None,
                            encoding='UTF-8')
    fig, ax = plt.subplots(len(course), len(course))
    fig.set_size_inches(19, 19)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for row in range(len(course)):
        for col in range(len(course)):
            if row != col:
                plt.sca(ax[row, col])
                scatter_plot(ax[row, col], course[row], course[col], dataset)
            else:
                plt.sca(ax[row, col])
                histogram(course[row], dataset, legend)
            if (row == len(course) - 1):
                plt.xlabel(course[col].replace('_', '\n'), wrap=True)
            if (col == 0):
                plt.ylabel(course[row].replace('_', '\n'), rotation='vertical')
    plt.legend(legend.keys(), loc=(40, 40))
    plt.show()
