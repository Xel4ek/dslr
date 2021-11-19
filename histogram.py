import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = np.genfromtxt('./resources/dataset_train.csv', delimiter=',', names=True, dtype=float, filling_values=None)
    course = ['Arithmancy','Astronomy', 'Herbology',
     'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies',
     'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
     'Care_of_Magical_Creatures', 'Charms', 'Flying']
    plt.legend(course)
    for item in course:
        plt.title(item)
        plt.hist(dataset[:][item], alpha=0.3)
        plt.show()
