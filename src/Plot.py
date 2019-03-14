import numpy as np
import pandas as pd
from sklearn import preprocessing
import helpers

#load data
data = pd.read_csv('./P1/PracticalOne/src/ENB2012_data.csv')
data_values = data.values

#extract data
x = pd.DataFrame(data_values[0:, 0:8])
y = pd.DataFrame(data_values[0:, 8:10])

#create scatter plots
fig_one = helpers.makeFigure()
fig_two = helpers.makeFigure()
for i in range(8):
    helpers.scatterPlotData(x.iloc[:,i], y.iloc[:,0], 'X'+str(i+1), 'Y1', i+1, fig_one)
    helpers.scatterPlotData(x.iloc[:,i], y.iloc[:,1], 'X'+str(i+1), 'Y2', i+1, fig_two)


#X Value Histograms
fig_three = helpers.makeFigure()
for i in range(8):
    helpers.histogram(x.iloc[:, i], fig_three, i+1, 'X'+str(i+1))

#Y Value Histograms
fig_four = helpers.makeFigure()    
for i in range(2):
    helpers.histogram(y.iloc[:, i], fig_four, i+1, 'Y'+str(i+1))

#Correlation Matrix for X
fig_five = helpers.makeFigure()
labels = ["X1", "X2","X3","X4","X5","X6","X7","X8"]
helpers.correlationMatrix(x, labels)
correlation = x.corr()
df = correlation.round(3)
print("Correlation Matrix for X")
print(df)

#Correlation Matrix for X and y
fig_six = helpers.makeFigure()
labels = ["X1", "X2","X3","X4","X5","X6","X7","X8", "Y1", "Y2"]
helpers.correlationMatrix(data, labels)
correlation = x.corr()
print("Correlation Matrix for X and y")
df = correlation.round(3)
print(df)

#show all the charts/graphs
helpers.show()


