import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dataset = np.genfromtxt('./resources/dataset_train.csv', delimiter=',',
                            names=True, filling_values=None, dtype=None,
                            encoding='UTF-8')
    course = ['Arithmancy', 'Astronomy', 'Herbology',
              'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies',
              'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
              'Care_of_Magical_Creatures', 'Charms', 'Flying']
    legend = {
        'Gryffindor': 'brown',
        'Hufflepuff': 'black',
        'Ravenclaw': 'steelblue',
        'Slytherin': 'green'}
    for item in course:
        plt.title(item)
        for house, color in legend.items():
            # data = np.extract(dataset[:][0] > 10)
            data = dataset[dataset['Hogwarts_House'] == house]
            # data = dataset[numpy.in1d(dataset[:][1], house)]
            plt.hist(data[:][item], alpha=0.7, label=house, color=color)
            plt.xlabel('Score')
            plt.ylabel('Number of students')
        plt.legend(legend.keys())
        plt.show()
