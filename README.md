# Energy Efficiency Optimization using Neural Networks
> This study looked into assessing the heating load and cooling load requirements of buildings (that is, energy efficiency) as a function of building parameters.
The project was done as part of Machine Learning class at the University of Texas at Dallas.
The entire summary of the project can be found in the [Jupyter Notebook](https://github.com/siddharthoza/Energy-Efficiency-Optimization-Neural-network/blob/master/Energy%20Efficiency%20Analysis%20.ipynb)

## Table of contents
* [General info](#general-info)
* [Technologies and Tools](#technologies-and-tools)
* [Setup](#setup)
* [Process](#process)
* [Code Examples](#code-examples)
* [Features](#features)
* [Status](#status)
* [Contact](#contact)

## General info

The goal of the study is to understand various machine learning classification algorithms and their performance. 
We have added the cooling load and heating load, which can define the overall load of the apartment. We have studied the trend of the overall load by dividing it into three classes. 

## Technologies and Tools
* Python 3.5
* TensorFlow

## Setup

The dataset used and its metadata can be found in the [dataset](https://github.com/siddharthoza/Energy-Efficiency-Optimization-Neural-network/tree/master/data). 
The jupyter notebook can be downloaded [here](https://github.com/siddharthoza/Energy-Efficiency-Optimization-Neural-network/blob/master/Energy%20Efficiency%20Analysis%20.ipynb) and can be used to reproduce the result. 
Installation of TensorFlow would be required to run all the models. 
You can find the instructions to install TensorFlow in [installation guide](https://www.tensorflow.org/install/pip).

## Process

* We have added the cooling load and heating load which will define the overall load of the apartment. 
* We studied the trend of overall load and divided it into three classes, low efficient, high efficient and average efficient.
* We first tried multioutput machine learning regressor models like KNN, logistic regression, random forest, decision tree, linear & kernelized SVM. 
* After that we created a neural network classification model to further increase the accuracy of the classification to 97%. 

## Code Examples
Some examples of usage:

````
# Linear SVM

param_grid = {"C": [1e0, 1e1, 1e2, 1e3, 1e4],"gamma": np.logspace(-2, 1, 2, 3, 5)}
grid_search_svm = MultiOutputRegressor(GridSearchCV(SVR(kernel='linear'), param_grid, cv=10,return_train_score=True))

grid_search_svm.fit(X_train, y_train)
train_r2_score=r2_score(y_train,grid_search_svm.predict(X_train))
test_r2_score=r2_score(y_test,grid_search_svm.predict(X_test))
output = output.append(pd.Series({'model':'SVM Linear', 'train_r2_score':train_r2_score,'test_r2_score':test_r2_score}),ignore_index=True )
output
````

````
# Bagging ensembler using Decision Tree Regressor as base model

from sklearn.ensemble import BaggingRegressor


param_grid = {'max_samples':[5,10], 'max_features':[1,2,3,4,5,6,7]}
bagging_DT = MultiOutputRegressor(GridSearchCV(BaggingRegressor(DecisionTreeRegressor(),n_estimators=750), 
                                                          param_grid,cv= 10,return_train_score=True))

bagging_DT.fit(X_train, y_train)
train_r2_score=r2_score(y_train,bagging_DT.predict(X_train))
test_r2_score=r2_score(y_test,bagging_DT.predict(X_test))
output = output.append(pd.Series({'model':'Multi Output DT Bagging', 'train_r2_score':train_r2_score,'test_r2_score':test_r2_score}),ignore_index=True )
output
````

````
#Ada Boosting on Linear SVM regressor¶

param =  { "n_estimators": [100,500,1000] }

base_svr=SVR(kernel='linear')
ada_svr = AdaBoostRegressor(base_estimator=base_svr,learning_rate = 0.7,random_state=10)
adaboost_svr = MultiOutputRegressor(GridSearchCV(ada_svr,param_grid=param,n_jobs=-1),n_jobs=-1)
adaboost_svr.fit(X_train, y_train)
train_r2_score=r2_score(y_train,adaboost_svr.predict(X_train))
test_r2_score=r2_score(y_test,adaboost_svr.predict(X_test))
output = output.append(pd.Series({'model':'Adaboost_LinearSVM', 'train_r2_score':train_r2_score,'test_r2_score':test_r2_score}),ignore_index=True )
output
````

````
# Neural Network Classification model

param_grid = {'epochs':[50, 200] , 'batch_size':[10, 50, 100]}

model = KerasClassifier(build_fn = model_classifier , verbose = 0)

grid_search_Keras_Class = GridSearchCV(model , param_grid , cv =10)

grid_search_Keras_Class.fit(X_train_class, y_train)

print('Best parameters for efficiency classification {}'.format(grid_search_Keras_Class.best_params_))

from sklearn.metrics import accuracy_score
print('The Train Accuracy score is',accuracy_score(y_train_class, grid_search_Keras_Class.predict(X_train_class)))
print('The Test Accuracy score is',accuracy_score(y_test_class, grid_search_Keras_Class.predict(X_test_class)))
````

## Features
The training and testing scores af various models are as listed below: 


sr no	| model	| train_r2_score	| test_r2_score
--- | --- | --- | ---|
0	| Linear Regressor	| 0.902539	| 0.899354
1	| KNN Regressor	| 0.935276	| 0.904540
2	| Random Forest Regressor	| 0.997856	| 0.984413
3	| SVM Linear	| 0.897939	| 0.896891
4	| SVM RBF	| 0.992589	| 0.986182
5	| Multi Output DT Bagging	| 0.875499	| 0.880045
6	| Adaboost_DecisionTree	| 0.999908	| 0.983499
7	| KNN Adaboost	| 0.972943	| 0.929250
8	| Adaboost_LinearSVM	| 0.896437	| 0.893522
9	| Gradient Boosting Regressor	| 0.999541	| 0.996787
10  | Logistic Reression | 0.875 | 0.880208
11  | Linear SVC | 0.875 | 0.880208
12  | SVM rbf | 0.970486 | 0.984375
13  | Random Forest Classifier | 1.0 | 0.96875
14  | Gradient Boosting Classifier | 1.0 | 0.979166
15  | Neural Network Regression model | 0.812056| 0.799832
16  | Neural Network Classification model  | 0.963541|  0.963541

## Status
Project is: _finished_

## Contact
Created by me and my teammate <a href="https://harshgupta.com/">Harsh Gupta</a>.

If you loved what you read here and feel like we can collaborate to produce some exciting stuff, or if you
just want to shoot a question, please feel free to connect with me on <a href="siddharth.oza@outlook.com" target="_blank">email</a>, 
<a href="www.linkedin.com/in/siddharthoza" target="_blank">LinkedIn</a>

My other projects can be found [here](www.siddharthoza.com).
