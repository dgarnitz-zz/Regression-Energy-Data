import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

data = pd.read_csv('./P1/PracticalOne/src/ENB2012_data.csv')

data_values = data.values
min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data_values)
normalized_data = pd.DataFrame(scaled_data)

#print(normalized_data)

x = normalized_data.iloc[0:, 0:8]
y = normalized_data.iloc[0:, 8:10] 

#Data loading was successful, as was data extraction
#you have two different Y values right now, so this was of doing regression probably wont work

#also the current MSE is extremely high, 9.322784
#This might be because the data is not currently normalized
#print(x)
print(y)

linreg = LinearRegression()
linreg.fit(x,y)
y_hat = linreg.predict(x)
print('MSE = ', mean_squared_error(y,y_hat)) 