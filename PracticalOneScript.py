#UNCLEAR which libraries need to be implemented
#you most likely will not use Numpy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = np.loadtxt('ENB2012_data.csv')

x = data[:, 1] #this DOES NOT WORK, need to review data to see how to slice
y = data[:, 2] #this DOES NOT WORK, need to review data to see how to slice

linreg = LinearRegression()
linreg.fit(x,y)
y_hat = linreg.predict(x)
print('MSE = ', mean_squared_error(y,y_hat))