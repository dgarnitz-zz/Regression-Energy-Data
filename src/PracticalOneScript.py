import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import helpers

data = pd.read_csv('./P1/PracticalOne/src/ENB2012_data.csv')

data_values = data.values
min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data_values)
normalized_data = pd.DataFrame(scaled_data)

x = normalized_data.iloc[0:, 0:8]
y = normalized_data.iloc[0:, 8:10]

#you have two different Y values right now ---> does SKLEARN know how to separate them? Do I need to do it manually?

fig_one = helpers.makeFigure()
fig_two = helpers.makeFigure()
for i in range(8):
    helpers.scatterPlotData(x.iloc[:,i], y.iloc[:,0], 'X'+str(i+1), 'Y1', i+1, fig_one)
    helpers.scatterPlotData(x.iloc[:,i], y.iloc[:,1], 'X'+str(i+1), 'Y2', i+1, fig_two)

helpers.show()

linreg = LinearRegression()
linreg.fit(x,y)
y_hat = linreg.predict(x)
print('MSE = ', mean_squared_error(y,y_hat))
