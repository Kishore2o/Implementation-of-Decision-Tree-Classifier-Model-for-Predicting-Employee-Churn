# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries
2. Upload and read the dataset.
3. .Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.kishore
RegisterNumber:  212222240050

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
X=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
Y=data["left"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
## data.head() :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/34dee842-e17b-4e0c-b725-06ed5b7dd49f)

## data.info() :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/ab2d881e-78cd-4777-baa4-786a1ee1eeae)

## isnull() and sum() :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/6b518de2-bb2e-4a99-9524-4fa94739f68f)

## data value counts() :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/fb53155c-4430-4d01-a205-7f935942fe93)

## data.head() for salary :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/0c6dd6cd-4dac-4812-87b0-c106103165dd)

## x.head() :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/692420f8-09fc-4a46-bb40-7754d36b72d4)

## accuracy value :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/3a2e5506-1819-4138-afaf-1b56c4625c60)

## data prediction :

![image](https://github.com/Kishore2o/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679883/4b6d539f-f9bf-4054-aa8d-6707aee830fe)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
