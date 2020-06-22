#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import keras
import os
import random
dataset=pd.read_csv("/root/deeplearning/train.csv")
dataset.head(10)

#To use data from the DataFrame for DL models, we convert the data into numpy arrays
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#print(X)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(16,input_dim=20,activation='relu'))
model.add(Dense(12,activation='relu'))
#model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_model=model.fit(X_train, y_train, epochs=10, batch_size=32)
model.summary()

model.save('/root/deeplearning/DL_model.h5')
model.save('/root/deeplearning/Iter_DL_model.h5')
accuracy = model.evaluate(X_test, y_test, verbose=0)
accuracy = accuracy[1]*100

from keras.models import load_model
layers_unit=8;epochs=50;batch_size=16;check=0
while accuracy < 94:
    old_accuracy=accuracy
    choice=random.randint(1,3)
    new_model = load_model('/root/deeplearning/Iter_DL_model.h5')
    new_model.pop()
    if choice==1:
        if epochs<=100:
            epochs=epochs+10
        else:
            check+=1
    elif choice==2:
        if batch_size<=64:
            batch_size=batch_size*2
        else:
            check+=1        
    elif choice==3:
        if len(new_model.layers)<=3:
            new_model.add(Dense(units=layers_unit,activation='relu'))
            layers_unit=layers_unit-4
        else:
            check+=1
        
        
    new_model.add(Dense(4,activation='softmax'))
    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    fit_model=new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,verbose=0)
    new_model.save('/root/deeplearning/Iter_DL_model.h5')
    accuracy = new_model.evaluate(X_test, y_test, verbose=0)
    accuracy = accuracy[1]*100
    print(" ")
    print("Accuracy found = "+str(accuracy))
    print(" ")
    if accuracy < 94 and check<3 and old_accuracy<accuracy:
        print("More modifications")
        print(" ")
        print(" ")
    else:
        if check==3:
            print("Randomisation stopped")
        elif accuracy > 94:
            print("Required accuracy achieved")
        elif accuracy<old_accuracy:
            print("Accuracy is decreasing. Stop training")
            
        file=open("/root/deeplearning/accuracy.txt","w")
        file.write("Accuracy of model is: "+str(accuracy))
        file.close()
        break
