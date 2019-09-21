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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, shuffle=True)

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
import xgboost

model1 = xgboost.XGBClassifier()

model2 = svm.SVC()

model3 = tree.DecisionTreeClassifier()

model4 = RandomForestClassifier()

from sklearn.metrics import accuracy_score, confusion_matrix
model1.fit(X, y)
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

y_pred1 = model1.predict(X_t)
y_pred2 = model2.predict(X_t)
y_pred3 = model3.predict(X_t)
y_pred4 = model4.predict(X_t)




