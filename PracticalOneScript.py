#UNCLEAR which libraries need to be implemented
#you most likely will not use Numpy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = np.loadtxt('ENB2012_data.csv') #this does not work, need to use Pandas

x = data[1:, 0:8]   #this DOES NOT WORK, need to review data to see how to slice
print(x)
y = data[1:, 8:10]  #this DOES NOT WORK, need to review data to see how to slice

linreg = LinearRegression()
linreg.fit(x,y)
y_hat = linreg.predict(x)
print('MSE = ', mean_squared_error(y,y_hat))