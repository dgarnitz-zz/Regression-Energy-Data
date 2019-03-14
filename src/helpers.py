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

def correlationMatrix(dataframe, labels): 
    corr = dataframe.corr()
    sns.heatmap(corr, annot=True, xticklabels=labels, yticklabels=labels, fmt='.3f')

def plotResults(y_actual, y_predicted, xlabel, ylabel, fig):
    ax = fig.add_subplot(111)
    
    ax.grid(color = 'lightgray', linestyle = '-', linewidth = 1)
    ax.set_axisbelow(True)

    ax.scatter(y_actual, y_predicted, color = 'red', alpha = .5, s = 140 , marker = '.')
    ax.plot(y_actual, y_actual, color = 'blue', alpha = .5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def show():
    plt.show()