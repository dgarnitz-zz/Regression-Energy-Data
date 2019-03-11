import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from pandas.io.formats.style import Styler
import helpers

#load data
data = pd.read_csv('./P1/PracticalOne/src/ENB2012_data.csv')
data_values = data.values

#extract data
x = pd.DataFrame(data_values[0:, 0:8])
y = pd.DataFrame(data_values[0:, 8:10])

#you have two different Y values right now ---> does SKLEARN know how to separate them? Do I need to do it manually?
        #According to Kasim it makes no difference mathematically, but its simpler to do them separately as you have 
        #singular parameter values for each output feature instead of a matrix of parameters per output feature

#create scatter plots
fig_one = helpers.makeFigure()
fig_two = helpers.makeFigure()
for i in range(8):
    helpers.scatterPlotData(x.iloc[:,i], y.iloc[:,0], 'X'+str(i+1), 'Y1', i+1, fig_one)
    helpers.scatterPlotData(x.iloc[:,i], y.iloc[:,1], 'X'+str(i+1), 'Y2', i+1, fig_two)

#helpers.show()                                         #commented out to facilitate developer - REMOVE AT THE END


#X Value Histograms
fig_three = helpers.makeFigure()
for i in range(8):
    helpers.histogram(x.iloc[:, i], fig_three, i+1, 'X'+str(i+1))

#Y Value Histograms
fig_four = helpers.makeFigure()    
for i in range(2):
    helpers.histogram(y.iloc[:, i], fig_four, i+1, 'Y'+str(i+1))

#helpers.show()                                         #commented out to facilitate developer - REMOVE AT THE END

#Correlation Matrix
fig_five = helpers.makeFigure()
helpers.correlationMatrix(x)
correlation = x.corr()
print(correlation)
#correlation.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
helpers.show()

#normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data_values)
normalized_data = pd.DataFrame(scaled_data)

x_normalized = normalized_data.iloc[0:, 0:8]
y_normalized = normalized_data.iloc[0:, 8:10]

#train the model & calculate the cost
linreg = LinearRegression()
linreg.fit(x,y)
y_hat = linreg.predict(x)
print('MSE = ', mean_squared_error(y,y_hat))
