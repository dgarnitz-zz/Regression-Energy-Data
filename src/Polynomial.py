import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline 
import helpers

def PolynomialRegression(x, y, output):
        #remove the training set 
        x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size = 0.2, random_state = 123) 

        linreg = LinearRegression()

        #standardize the data
        scaler = preprocessing.StandardScaler().fit(x_training)

        #create pipeline & grid
        pipeline = Pipeline([('scaler', scaler), 
                ('polynomial', PolynomialFeatures()),
                ('model', linreg)])

        grid = {'polynomial__degree': range(1,6),
                'polynomial__include_bias': ["False"]} 

        #cross validation
        clf = GridSearchCV(pipeline, param_grid = grid, cv=5, refit = True)

        #fit and tune the model
        clf.fit(x_training, y_training)

        #refit on the entire training set
        clf.refit

        #generalize on a new dataset
        y_prediction = clf.predict(x_testing)
        print("MSE: ")
        print (mean_squared_error(y_testing, y_prediction))
        print("MAE: ")
        print(mean_absolute_error(y_testing, y_prediction))
        print("R2 of the model: ")
        print(r2_score(y_testing, y_prediction))
        
        #visualize the results
        fig = helpers.makeFigure()
        xlabel = "Actual " + output 
        ylabel = "Generalized " + output 
        helpers.plotResults(y_testing, y_prediction, xlabel, ylabel, fig)
        


#load data
data = pd.read_csv('./P1/PracticalOne/src/ENB2012_data.csv')
data_values = data.values

#extract data
X = pd.DataFrame(data_values[0:, 0:8])
y1 = pd.DataFrame(data_values[0:, 8:9])
y2 = pd.DataFrame(data_values[0:, 9:10])

#run the regression for each output feature
PolynomialRegression(X, y1, "Heating Load")
PolynomialRegression(X, y2, "Cooling Load")

#helpers.show()