import matplotlib.pyplot as plt
import numpy as np


def histogram(course, dataset):
    legend = {
        'Gryffindor': 'brown',
        'Hufflepuff': 'black',
        'Ravenclaw': 'steelblue',
        'Slytherin': 'green'}
    plt.title(course)
    for house, color in legend.items():
        data = dataset[dataset['Hogwarts_House'] == house]
        plt.hist(data[:][course], alpha=0.7, label=house, color=color)
        plt.xlabel('Score')
        plt.ylabel('Number of students')
    plt.legend(legend.keys())


if __name__ == '__main__':
    dataset = np.genfromtxt('./resources/dataset_train.csv', delimiter=',',
                            names=True, filling_values=None, dtype=None,
                            encoding='UTF-8')
    courses = ['Arithmancy', 'Astronomy', 'Herbology',
               'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies',
               'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
               'Care_of_Magical_Creatures', 'Charms', 'Flying']

    for item in courses:
        histogram(item, dataset)
        plt.show()
