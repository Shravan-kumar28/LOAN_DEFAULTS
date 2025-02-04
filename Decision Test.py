# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:40:54 2025

@author: shrav
"""
"""
Decision Tree

1.	A cloth manufacturing company is interested to know about the different attributes
 contributing to high sales. Build a decision tree & random forest model with Sales as t
 he target variable (first convert it into categorical variable).
 
 Data Dictionary:

RI	:Refractive Index
Na	:Sodium content in the glass
Mg	:Magnesium content in the glass
Al	:Aluminum content in the glass
Si	:Silicon content in the glass
K	:Potassium content in the glass
Ca	:Calcium content in the glass

Ba	:Barium content in the glass
Fe	:Iron content in the glass

Type	:Type of glass classification

"""


#Importing required libraries
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
import sklearn.metrics as skmet
from sklearn.model_selection import GridSearchCV


#Load dataset
df=pd.read_csv(r"C:\Users\shrav\Downloads\Data Set DT\ClothCompany_Data.csv")


df.info()
# Checking for Null values
df.isnull().sum()
des=df.describe()

#Data discretization
df['Sales']=pd.cut(df['Sales'],bins=2,labels=['low','high'])

# Target variable categories
df['Sales'].unique()
df['Sales'].value_counts()

# Data split into Input and Output
Sales_x=df.iloc[:,1:12] # Predictors 
Sales_y=df['Sales'] # Target

# #### Separating Numeric and Non-Numeric columns
Catgorical_features=Sales_x.select_dtypes(include=['object']).columns
Catgorical_features

Numerical=Sales_x.select_dtypes(exclude=['object']).columns
Numerical


# ### Data Preprocessing
# ### MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1
# ### Encoding - One Hot Encoder to convert Categorical data to Numeric values
Process1=Pipeline([('Scale',MinMaxScaler())])

Process2=Pipeline([('onehot',OneHotEncoder())])

# Creating a transformation of variable with ColumnTransformer()
ColumTrans=ColumnTransformer(transformers=[('num',Process1,Numerical),
                                           ('cat',Process2,Catgorical_features)])

SclOne=ColumTrans.fit(Sales_x)

joblib.dump(SclOne,'Scalar_Onehot')                                            

CleanData=pd.DataFrame(SclOne.transform(Sales_x),columns=SclOne.get_feature_names_out())

CleanData.iloc[:,0:7].columns

# Note: If you get any error then update the scikit-learn library version & restart the kernel to fix it

# ### Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True
CleanData.iloc[:, 0:7].plot(kind='box',subplots=True,sharey=False,figsize=(18,6))
plt.subplots_adjust(wspace=0.75)
plt.show()

# #### Outlier analysis
winsorizer=Winsorizer(capping_method='iqr',
                      fold=1.5,
                      tail='both',
                      variables=['num__CompPrice','num__Price'])

Outliers=winsorizer.fit(CleanData[['num__CompPrice','num__Price']])

CleanData[['num__CompPrice','num__Price']]=Outliers.transform(CleanData[['num__CompPrice','num__Price']])

# Clean data
# Verify for outliers
CleanData.iloc[:, 0:7].plot(kind='box',subplots=True,sharey=False,figsize=(18,6))
plt.subplots_adjust(wspace=0.75)
plt.show()

# Split data into train and test with Stratified Sample technique
# from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(CleanData,Sales_y,test_size=0.2,stratify=Sales_y,random_state=0)

# Proportion of Target variable categories are consistent across train and test
print(Y_train.value_counts()/800)
print("\n")
print(Y_test.value_counts()/200)

### Decision Tree Model
DTree=DT(criterion='entropy')
DTree.fit(X_train,Y_train)

# Prediction on Test Data
pred=DTree.predict(X_test)
pred

# Accuracy
print(skmet.accuracy_score(Y_test,pred))

pd.crosstab(Y_test,pred,rownames=['Actual'],colnames=['Predictors'])

### Hyperparameter Optimization
# create a dictionary of all hyperparameters to be experimented
param_grid = { 'criterion':['gini', 'entropy'], 'max_depth': np.arange(3, 15)}

# Decision tree model
model=DT()

# GridsearchCV with cross-validation to perform experiments with parameters set
Dtree_grid=GridSearchCV(model, param_grid,cv=5,scoring='accuracy',return_train_score=False,verbose=1)
# Train the model with Grid search optimization technique
Dtree_grid.fit(X_train,Y_train)

# The best set of parameter values
Dtree_grid.best_params_

# Model with best parameter values
Dtree_best=Dtree_grid.best_estimator_

# Prediction on Test Data
pred_new=Dtree_best.predict(X_test)
pred_new

# Accuracy
print(skmet.accuracy_score(Y_test,pred_new))

# Model evaluation
# Cross Table (Confusion Matrix)
pd.crosstab(Y_test,pred_new,rownames=['Actual'], colnames=['Predictors'])

#####################
# Generate Tree visualization

import os
os.environ['PATH'] += os.pathsep + r"C:\Users\shrav\AppData\Local\Programs\Python\Python313\Lib\site-packages\Graphviz-12.2.1-win64\bin"

import graphviz
from sklearn import tree

predictors=list(CleanData.columns)


dot_data=tree.export_graphviz(Dtree_best,filled=True,
                              rounded=True,
                              feature_names=predictors,
                              class_names=['low','high'],
                              out_file=None
                              )

graph=graphviz.Source(dot_data)
graph

#####################

# Prediction on Train Data
pred_train=Dtree_best.predict(X_train)
pred_train

# Accuracy
print(skmet.accuracy_score(Y_train,pred_train))

# Confusion Matrix
pd.crosstab(Y_train,pred_train,rownames=["actual"],colnames=["Predictors"])

## Model Training with Cross Validation
from sklearn.model_selection import cross_validate

def cross_validation(model,_X,_y,_cv=5):
    '''
    Function to perform 5 Folds Cross-Validation
    Parameters
    ----------
    model: Python Class, default=None
          This is the machine learning algorithm to be used for training.
    _X: array
          This is the matrix of features.
    _y: array
          This is the target variable.
    _cv: int, default=5
          Determines the number of folds for cross-validation.
    Returns
    -------
    The function returns a dictionary containing the metrics 'accuracy', 'precision',
    'recall', 'f1' for both training set and validation set.
    '''
    _scoring=['accuracy','precision','recall','f1']
    results=cross_validate(estimator=model,X=_X,y=_y,
                           cv=_cv,scoring=_scoring,
                           return_train_score=True)
    
    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })

labelEnCoder=LabelEncoder()
encoder_y=labelEnCoder.fit_transform(Y_train)

decision_results=cross_validation(Dtree_best,X_train,encoder_y,5)
decision_results


def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''       
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.4, color = 'blue', label = 'Training')
        plt.bar(X_axis + 0.2, val_data, 0.4, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize = 30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize = 14)
        plt.ylabel(y_label, fontsize = 14)
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
model_name="decision tree"
plot_result(model, 'Accuracy','Accuracy in 5 folds',
            decision_results['Training Accuracy scores'],
            decision_results['Validation Accuracy scores'])