# Logistic Regression

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv("Trainset.csv")
dataset.head()

# Creating Correlation Matrix
corrMat = dataset.corr()

# Handling missing data
mean1 = dataset['Homepage'].mean()
dataset['Homepage'].fillna(mean1, inplace=True)
mean1 = dataset['Homepage _Duration'].mean()
dataset['Homepage _Duration'].fillna(mean1, inplace=True)
mean2 = dataset['Aboutus'].mean()
dataset['Aboutus'].fillna(mean2, inplace=True)
mean2 = dataset['Aboutus_Duration'].mean()
dataset['Aboutus_Duration'].fillna(mean2, inplace=True)
mean3 = dataset['Contactus'].mean()
dataset['Contactus'].fillna(mean3, inplace=True)
mean3 = dataset['Contactus_Duration'].mean()
dataset['Contactus_Duration'].fillna(mean3, inplace=True)
mean4 = dataset['BounceRates'].mean()
dataset['BounceRates'].fillna(mean4, inplace=True)
mean4 = dataset['ExitRates'].mean()
dataset['ExitRates'].fillna(mean4, inplace=True)

dataset.info()

print(dataset.groupby('Month')['Revenue'].mean().sort_values())
print(dataset.groupby('Weekend')['Revenue'].mean().sort_values())
print(dataset.groupby('VisitorType')['Revenue'].mean().sort_values())
print(dataset.groupby('OperatingSystems')['Revenue'].mean().sort_values())
print(dataset.groupby('Browser')['Revenue'].mean().sort_values())
print(dataset.groupby('Province')['Revenue'].mean().sort_values())

# Handling Catagorical Data
dataset = pd.get_dummies(dataset, columns=['Month'])
dataset['Weekend'] = dataset['Weekend'].astype(int)
dataset = pd.get_dummies(dataset, columns=['OperatingSystems'])
dataset = pd.get_dummies(dataset, columns=['Browser'])
dataset = pd.get_dummies(dataset, columns=['Province'])
dataset = pd.get_dummies(dataset, columns=['VisitorType'])

# Removing additional dependancy
del dataset['Month_Aug']
del dataset['OperatingSystems_1']
del dataset['Browser_9']
del dataset['Province_1']
del dataset['VisitorType_Other']

# Getting X and y
y = dataset.iloc[:, 12].values
del dataset['Revenue']
dataset = dataset.astype(float)
X = dataset.iloc[:, :].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=True)

## Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X, y)
#
#y_pred = classifier.predict(X)
#
#from sklearn.metrics import accuracy_score
#accuracy_score(y, y_pred)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
#import xgboost

#model1 = xgboost.XGBClassifier()

model2 = svm.SVC()

model3 = tree.DecisionTreeClassifier()

model4 = RandomForestClassifier()

from sklearn.metrics import accuracy_score, confusion_matrix
#model1.fit(X, y)



#y_pred= model1.predict(X_test)
#acc = accuracy_score(y_test, y_pred)
#print("Accuracy of %s is %s"%(model1, acc))
#cm = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix of %s is %s"%(model1, cm))

model2.fit(X, y)
#y_pred= model2.predict(X_test)
#acc = accuracy_score(y_test, y_pred)
#print("Accuracy of %s is %s"%(model2, acc))
#cm = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix of %s is %s"%(model2, cm))

model3.fit(X, y)
#y_pred= model3.predict(X_test)
#acc = accuracy_score(y_test, y_pred)
#print("Accuracy of %s is %s"%(model3, acc))
#cm = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix of %s is %s"%(model3, cm))

model4.fit(X, y)
#y_pred= model4.predict(X_test)
#acc = accuracy_score(y_test, y_pred)
#print("Accuracy of %s is %s"%(model4, acc))
#cm = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix of %s is %s"%(model4, cm))
    
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu', input_dim = 50))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 1000)

#y_pred_ann = classifier.predict(X_test)

#y_pred_ann = (y_pred_ann > 0.5)

## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred_ann)



from keras.wrappers.scikit_learn import KerasClassifier

def c_model():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def new_model():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu', input_dim = 50))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

model = KerasClassifier(build_fn=new_model, epochs=50, batch_size=32)

model.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

model = KerasClassifier(build_fn=new_model)

batch_sizes = [10, 20, 50, 100]
epochs = [5, 10, 50]
parameters = {'batch_size': batch_sizes, 'epochs': epochs}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)

print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)

model.fit(X, y)

def new_model(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu', input_dim = 50))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

model = KerasClassifier(build_fn=new_model, epochs=50, batch_size=32)
parameters = {'optimizer':['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}
clf1 = GridSearchCV(model, parameters)
clf1.fit(X_train, y_train)

print(clf1.best_score_, clf.best_params_)
means = clf1.cv_results_['mean_test_score']
parameters = clf1.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)


def new_model(activation):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 25, init = 'uniform', activation = activation, input_dim = 50))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = activation))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

model = KerasClassifier(build_fn=new_model, epochs=50, batch_size=32)
parameters = {'activation':['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']}
clf2 = GridSearchCV(model, parameters)
clf2.fit(X_train, y_train)

print(clf2.best_score_, clf.best_params_)
means = clf2.cv_results_['mean_test_score']
parameters = clf2.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)


# For self evaluation
classifier = Sequential()
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'sigmoid', input_dim = 50))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'Adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 50)

y_pred_ann = classifier.predict(X_test)

y_pred_ann = (y_pred_ann > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_ann)

from keras import backend as K
K.clear_session()
accuracy



#For final predication
classifier = Sequential()
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'sigmoid', input_dim = 50))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['auc'])
classifier.fit(X, y, batch_size = 20, nb_epoch = 50)




import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args






















testset = pd.read_csv("xtest.csv")
del testset['ID']

mean1 = testset['Homepage'].mean()
testset['Homepage'].fillna(mean1, inplace=True)
mean1 = testset['Homepage _Duration'].mean()
testset['Homepage _Duration'].fillna(mean1, inplace=True)
mean2 = testset['Aboutus'].mean()
testset['Aboutus'].fillna(mean2, inplace=True)
mean2 = testset['Aboutus_Duration'].mean()
testset['Aboutus_Duration'].fillna(mean2, inplace=True)
mean3 = testset['Contactus'].mean()
testset['Contactus'].fillna(mean3, inplace=True)
mean3 = testset['Contactus_Duration'].mean()
testset['Contactus_Duration'].fillna(mean3, inplace=True)
mean4 = testset['BounceRates'].mean()
testset['BounceRates'].fillna(mean4, inplace=True)
mean4 = testset['ExitRates'].mean()
testset['ExitRates'].fillna(mean4, inplace=True)

testset.info()

testset = pd.get_dummies(testset, columns=['Month'])
testset['Weekend'] = testset['Weekend'].astype(int)
testset = pd.get_dummies(testset, columns=['OperatingSystems'])
testset = pd.get_dummies(testset, columns=['Browser'])
testset = pd.get_dummies(testset, columns=['Province'])
testset = pd.get_dummies(testset, columns=['VisitorType'])

del testset['Month_Aug']
del testset['OperatingSystems_1']
#del testset['Browser_1']
del testset['Province_1']
del testset['VisitorType_Other']

X_t = testset.iloc[:, :].values

#y_pred1 = model1.predict(X_t)
y_pred2 = model2.predict(X_t)
y_pred3 = model3.predict(X_t)
y_pred4 = model4.predict(X_t)

Sub_predictions = classifier.predict(X_t)
#Sub_predictions.dtype
#print(testset)

submissions=pd.DataFrame({"Revenue": Sub_predictions[:,0]})
submissions['Revenueed'] = np.where(submissions['Revenue']>0.5, 1, 0)
submissions.to_csv("sample_submission2.csv", index=False, header=True)


