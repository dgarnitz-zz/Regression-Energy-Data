import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
# correlation = x.corr()
# df = correlation.round(3)
# print(df)
helpers.show()

#normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data_values)
normalized_data = pd.DataFrame(scaled_data)

x_normalized = normalized_data.iloc[0:, 0:8]
#y_normalized = normalized_data.iloc[0:, 8:10]

#Remove the training set 
x_training, x_testing, y_training, y_testing = train_test_split(x_normalized, y, test_size = 0.2, random_state = 0)

#K Fold Cross Validation (K=5) & Training
x_training, x_validation, y_training, y_valiation = train_test_split(x_training, y_training, test_size = 0.2, random_state = 0)
linreg = LinearRegression()
scores = cross_val_score(linreg, x_training, y_training, scoring = 'neg_mean_absolute_error' ,cv=5)
print(scores)

#train the model & calculate the cost
linreg.fit(x,y)
y_hat = linreg.predict(x_normalized)
print('MSE = ', mean_squared_error(y,y_hat))
