import matplotlib.pyplot as plt
import seaborn as sns

def makeFigure():
    return plt.figure()

def scatterPlotData(x, y, xlabel, ylabel, position, fig):
    ax = fig.add_subplot(330+position)
    
    ax.grid(color = 'lightgray', linestyle = '-', linewidth = 1)
    ax.set_axisbelow(True)

    ax.scatter(x, y, color = 'blue', alpha = .2, s = 140 , marker = '.')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def histogram(x, fig, position, label):
    ax = fig.add_subplot(330+position)
    n, bins, patches = plt.hist(x, density = 0, facecolor = 'green', alpha = .2)
    plt.xlabel(label)
    plt.ylabel('Frequency')

    
def correlationMatrix(dataframe):
    corr = dataframe.corr()
    sns.heatmap(corr, annot=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values, fmt='.3f')


def show():
    plt.show()