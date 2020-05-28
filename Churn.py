import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Heroku-Demo-master\\Churn")

FullRaw = pd.read_csv("Churn_Modelling.csv")

FullRaw.drop(['RowNumber','CustomerId','Surname'], axis =1, inplace =True)

FullRaw.isnull().sum()

Category_Vars = (FullRaw.dtypes == 'object')
dummyDf = pd.get_dummies(FullRaw.loc[:,Category_Vars],drop_first =True)

FullRaw2 = pd.concat([FullRaw.loc[:,~Category_Vars],dummyDf], axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw2,test_size =0.3, random_state =123)

Train_X = Train.drop(['Exited'], axis =1)
Train_Y = Train['Exited'].copy()
Test_X =Test.drop(['Exited'], axis =1)
Test_Y = Test['Exited'].copy()

from sklearn.preprocessing import StandardScaler

Train_scaling = StandardScaler().fit(Train_X)
Train_X_Std = Train_scaling.transform(Train_X)
Test_X_Std = Train_scaling.transform(Test_X)

Train_X_Std =pd.DataFrame(Train_X_Std, columns = Train_X.columns)
Test_X_Std = pd.DataFrame(Test_X_Std, columns =Test_X.columns)


import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU,ReLU

Classifier = Sequential()
Classifier.add(Dense(units= 6, kernel_initializer ='he_uniform', activation ='relu',input_dim = 11))
Classifier.add(Dense(units= 6, kernel_initializer ='he_uniform', activation ='relu'))
Classifier.add(Dense(units = 1,kernel_initializer = 'glorot_uniform', activation ='sigmoid'))
Classifier.compile(optimizer = 'Adamax',loss = 'binary_crossentropy', metrics = ['accuracy'])

model = Classifier.fit(Train_X_Std,Train_Y,batch_size = 10, epochs = 100, validation_split =0.3)

Test_pred = Classifier.predict(Test_X_Std)

Test['Test_prob']= Test_pred
Test['Test_Class'] = np.where(Test['Test_prob'] > 0.5,1,0)

Test['Test_Class'].value_counts()

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test['Test_Class'], Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

import pickle

pickle.dump(Classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
arr = np.array([[1.33,-0.46,-0.0069,-1.219,0.798,-1.54,0.969,0.204,-0.575,
                      1.73,-1.100]])
print(model.predict(arr))

#from keras.layers import Dropout, Activation
#from keras.wrappers.scikit_learn import KerasClassifier
#
#def create_model(layers, activation):
#    model = Sequential()
#    for i,nodes in enumerate(layers):
#        if i == 0:
#            model.add(Dense(nodes, input_dim = Train_X.shape[1]))
#            model.add(Activation(activation))
#            model.add(Dropout(0.3))
#            
#        else:
#            model.add(Dense(nodes))
#            model.add(Activation(activation))
#            model.add(Dropout(0.3))
#            
#    model.add(Dense(units = 1, kernel_initializer ='glorot_uniform',activation = 'sigmoid'))
#    model.compile(optimizer = 'Adamax',loss = 'binary_crossentropy', metrics = ['accuracy'])
#    
#    return(model)
#
#model = KerasClassifier(build_fn = create_model, verbose = 0)
#
#from sklearn.model_selection import GridSearchCV
#
#my_layers = [[40],[20,40]]
#my_activation = ['sigmoid','relu']
#my_param_grid = {'layers': my_layers, 'activation': my_activation, 'epochs': [50,100],
#                 'batch_size': [10,20,30]}
#
#Grid = GridSearchCV(estimator = model,param_grid = my_param_grid, 
#                    scoring = 'accuracy',cv = 5).fit(Train_X_Std,Train_Y)
#
