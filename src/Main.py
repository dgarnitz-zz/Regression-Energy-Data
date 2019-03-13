import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline 
import helpers

#load data
data = pd.read_csv('./P1/PracticalOne/src/ENB2012_data.csv')
data_values = data.values

#extract data
x = pd.DataFrame(data_values[0:, 0:8])
y = pd.DataFrame(data_values[0:, 8:10])

#normalize the data
# min_max_scaler = preprocessing.MinMaxScaler()
# scaled_data = min_max_scaler.fit_transform(data_values)
# normalized_data = pd.DataFrame(scaled_data)

#x_normalized = normalized_data.iloc[0:, 0:8]
#y_normalized = normalized_data.iloc[0:, 8:10]

#Remove the training set 
x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size = 0.2, random_state = 123) 

linreg = LinearRegression()

scaler = preprocessing.StandardScaler().fit(x_training)
pipeline = Pipeline([('scaler', scaler), 
        ('polynomial', PolynomialFeatures()),
        ('model', linreg)])

grid = [{'polynomial__degree': [1]}] #would use range(1,x) for a degree of x

#cross validation
clf = GridSearchCV(pipeline, param_grid = grid, cv=10, refit = True)

#fit and tune the model
clf.fit(x_training, y_training)

#refit on the entire training set
clf.refit

#generalize on a new dataset
y_prediction = clf.predict(x_testing)
print (mean_squared_error(y_testing, y_prediction))
print(r2_score(y_testing, y_prediction))
