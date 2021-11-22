import matplotlib.pyplot as plt
import numpy as np


def histogram(course, dataset, legend):
    for house, color in legend.items():
        data = dataset[dataset['Hogwarts_House'] == house]
        plt.hist(data[:][course], alpha=0.7, label=house, color=color)


if __name__ == '__main__':
    legend = {
        'Gryffindor': 'brown',
        'Hufflepuff': 'black',
        'Ravenclaw': 'steelblue',
        'Slytherin': 'green'
    }
    dataset = np.genfromtxt('./resources/dataset_train.csv', delimiter=',',
                            names=True, filling_values=None, dtype=None,
                            encoding='UTF-8')
    courses = ['Arithmancy', 'Astronomy', 'Herbology',
               'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies',
               'Ancient_Runes', 'History_of_Magic', 'Transfiguration',
               'Potions', 'Care_of_Magical_Creatures', 'Charms', 'Flying']
    histogram(courses[0], dataset, legend)
    plt.xlabel('Score')
    plt.ylabel('Number of students')
    plt.title(courses[0])
    plt.legend(legend.keys())
    plt.show()
