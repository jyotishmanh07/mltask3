#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import sklearn
import joblib

dataset  =  pd.read_csv('/root/machinelearning/titanic_train.csv')
#dataset.head(20)
#dataset.info()
#dataset.columns

#Performing data visualization to clean and understand data using Seaborn
#import seaborn as sns
#sns.set()
#gender = dataset['Sex']

# bar graph
#sns.countplot(gender)
#sns.countplot(dataset['Survived'], hue='Sex', data=dataset)
#sns.countplot(dataset['Survived'], hue='Pclass', data=dataset)

#age = dataset['Age']
#sns.distplot(age)
#print(dataset.isnull())

#sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')
#sns.distplot(age.dropna() ,bins=40)
#sns.countplot(dataset['SibSp'], data=dataset, hue='Survived')
#sns.boxplot(data=dataset, y='Age' , x='Pclass')

def lw(cols):
    age = cols[0]
    Pclass = cols[1]
    if pd.isnull(age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        elif Pclass == 3:
            return 25
        else:
            return 30
    else:
        return age
    

dataset['Age'] = dataset[['Age', 'Pclass']].apply(lw , axis=1)

#sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')
dataset.drop('Cabin', axis=1, inplace=True )

#sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')

# univariate : histogram : frequency distribution
#fare = dataset['Fare']

#fare.hist(bins=50, color='red', figsize=(5,1) )

y = dataset['Survived']
X = dataset[ ['Pclass','Sex', 'Age', 'SibSp', 'Parch' , 'Embarked' ]]


# Encoding all Categorical Variables
sex = dataset['Sex']
sex = pd.get_dummies(sex, drop_first=True)

pclass = dataset['Pclass']
pclass = pd.get_dummies(pclass, drop_first=True)

sibsp = dataset['SibSp']
sibsp = pd.get_dummies(sibsp, drop_first=True)

parch = dataset['Parch']
parch = pd.get_dummies(parch, drop_first=True)

embarked = dataset['Embarked']
embarked = pd.get_dummies(embarked, drop_first=True)

age = dataset[ 'Age']
X = pd.concat([age/max(age), embarked, parch, sibsp, pclass, sex] ,  axis=1)
print(X)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
test_size=0.1;accuracy=0
while accuracy<85:
    old_accuracy=accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    con=confusion_matrix(y_test , y_pred)
    record = con[0][0]+con[0][1]+con[1][0]+con[1][1]
    trueanswer = con[0][0] + con[1][1]
    accuracy = trueanswer / record * 100
    test_size=test_size+0.1
    if accuracy<85 and old_accuracy<accuracy:
        print("Accuracy is "+str(accuracy))
        print("More training")
        print(" ")
        print(" ")
    elif test_size==0.70:
        print("Finished training")
        file=open("/root/machinelearning/accuracy.txt","w")
        file.write("Accuracy of model is: "+str(accuracy))
        file.close()
        break
        break
    elif old_accuracy>accuracy:
        print("Accuracy is decreasing. Stop training")
        accuracy=old_accuracy
        file=open("/root/machinelearning/accuracy.txt","w")
        file.write("Accuracy of model is: "+str(accuracy))
        file.close()
        break
        break
