""" This dataset sourced from the UCI Machine Learning Repository entails the development of a predictive model aimed at
classifying the flower species based on the provided features."""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


columns=["sepal width",'petal length',"petal width","class"]
df=pd.read_csv("D:\Data Science\Pandas\iris.data",names=columns)
# for online acess : pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",names=columns)

# combined dataset
Flowers=pd.DataFrame(df)

#individual dataset
Flowers_setosa=Flowers[Flowers["class"]=="Iris-setosa"]
Flowers_versicolor=Flowers[Flowers["class"]=="Iris-versicolor"]
Flowers_virginica=Flowers[Flowers["class"]=="Iris-virginica"]

# description comparision of all 3 classes
# print(Flowers_setosa["sepal width"].describe(),"\n\n",Flowers_versicolor["sepal width"].describe(),"\n\n",Flowers_virginica["sepal width"].describe())


#The DecisionTreeRegressor exclusively yields numerical values as output. Consequently, we initiate the process by converting our classes into numerical values utilizing the map function.
"""
mapping is 
Iris-setosa-0
Iris-versicolor-1
Iris-virginica-2
"""
maping={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
df["class"]=df["class"].map(maping)

#initialising the parameters and output
y=df["class"]
parameters=["sepal width","petal length","petal width"]
X=df[parameters]

#created the model
fl_model=DecisionTreeRegressor(random_state=1)

#splitting the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,train_size=0.5,random_state=1)

#now we will fit the model with the training data
fl_model.fit(X_train,y_train)

#now we will see the predictions of the model
predictions=fl_model.predict(X_test)

#calculating the accuracy of our decision tree regressor model
accuracy_of_decisiontreeregressor=accuracy_score(y_test,predictions)

#comparing the outputs with the actual data
results=pd.DataFrame({"Actual":y_test,"Predicted":predictions})
# print(results)
# print(accuracy_of_decisiontreeregressor) - 97.33% accuracy

"""Our analysis reveals that our model achieves an accuracy of 97.33% in predicting the flower species based on the provided features. Additionally, we plan to evaluate an SVM (Support Vector Machine) model to determine if it yields superior results compared to the decision tree model."""


#creating the model
fl_model_svm=svm.SVC(kernel="linear")

#fitting it with the same data
fl_model_svm.fit(X_train,y_train)
#predictions based on X_test
predictions2=fl_model_svm.predict(X_test)

#accuracy
accuracy_of_svm=accuracy_score(y_test,predictions)

#comparing the outputs with the actual data\
results2=pd.DataFrame({"Actual":y_test,"Predicted":predictions2})
# print(results2)
# print(accuracy_of_svm) - 97.33%

"""It appears that both models are yielding identical accuracy rates."""
