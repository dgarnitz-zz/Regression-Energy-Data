import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('ENB2012_data.csv')

x = data.iloc[1:, 0:8]
y = data.iloc[1:, 8:10] 

#Data loading was successful, as was data extraction
#you have two different Y values right now, so this was of doing regression probably wont work

#also the current MSE is extremely high, 9.322784
#This might be because the data is not currently normalized

linreg = LinearRegression()
linreg.fit(x,y)
y_hat = linreg.predict(x)
print('MSE = ', mean_squared_error(y,y_hat)) 