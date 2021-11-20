import numpy as np
from matplotlib import pyplot as plt


def scatter_plot(course_x, course_y, input_dataset):

    legend = {
        'F-Test': '',
        'Gryffindor': 'brown',
        'Hufflepuff': 'black',
        'Ravenclaw': 'steelblue',
        'Slytherin': 'green'}
    for house, color in legend.items():
        if house == 'F-Test':
            continue
        data = input_dataset[input_dataset['Hogwarts_House'] == house][[course_x, course_y]]
        data = data[~np.isnan(data[course_x]) & ~np.isnan(data[course_y])]
        x = data[course_x]
        x = x[~np.isnan(x)]
        y = data[course_y]
        y = y[~np.isnan(y)]
        plt.scatter(x, y, alpha=0.7, label=house, color=color)
        plt.xlabel(course_x)
        plt.ylabel(course_y)

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
    fig, ax = plt.subplots(len(course), len(course))
    fig.set_size_inches(100, 100)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(len(course)):
        for j in range(len(course)):
            if i != j:
                plt.sca(ax[j, i])
                scatter_plot(course[i], course[j], dataset)
            else:
                ax[j, i].text(0.5, 0.5,'later', fontsize=18, ha='center')
