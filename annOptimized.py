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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X=sc.fit_transform(X)
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adadelta,Adamax,Adagrad,RMSprop,Nadam


# Part 3 - Making the predictions and evaluating the model
#Accuracy: 90.65%
#learning rate: 8.6e-05
#num_dense_layers: 5
num_dense_nodes = 49
#activation: softsign
#optimizer: Adamax
#epochs: 387
#batch_size: 118



#Accuracy: 88.98%
learning_rate= 1.3e-05
num_dense_layers= 12
num_dense_nodes= 320
activation= 'tanh'
optimizer='Nadam'
epochs=162
batch_size= 367

if(optimizer=='Adam'):
        optimizered = Adam(lr=learning_rate)
elif(optimizer=='SGD'):
    optimizered = SGD(lr=learning_rate)
elif(optimizer=='RMSprop'):
    optimizered = RMSprop(lr=learning_rate)
elif(optimizer=='Adagrad'):
    optimizered = Adagrad(lr=learning_rate)
elif(optimizer=='Adadelta'):
    optimizered = Adadelta(lr=learning_rate)
elif(optimizer=='Adamax'):
    optimizered = Adamax(lr=learning_rate)
elif(optimizer=='Nadam'):
    optimizered = Nadam(lr=learning_rate)


classifier1=Sequential()
classifier1.add(Dense(num_dense_nodes, activation = activation, input_dim = 50))

for i in range(num_dense_layers):
    classifier1.add(Dense( num_dense_nodes,activation = activation))

classifier1.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier1.compile(optimizer = Nadam(lr=learning_rate) , loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier1.fit(X_test, y_test, batch_size = batch_size, nb_epoch = epochs)








# Predicting the Test set results
y_pred = classifier1.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#
#
#


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
X_t=sc.fit_transform(X_t)

Sub_predictions = classifier1.predict(X_t)
#Sub_predictions.dtype
#print(testset)

submissions=pd.DataFrame({"Revenue": Sub_predictions[:,0]})
submissions['Revenueed'] = np.where(submissions['Revenue']>0.5, 1, 0)
submissions.to_csv("sample_submission5.csv", index=False, header=True)

