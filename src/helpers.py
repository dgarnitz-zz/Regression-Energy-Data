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
    labels = ["X1", "X2","X3","X4","X5","X6","X7","X8"]
    sns.heatmap(corr, annot=True, xticklabels=labels, yticklabels=labels, fmt='.3f')


def show():
    plt.show()