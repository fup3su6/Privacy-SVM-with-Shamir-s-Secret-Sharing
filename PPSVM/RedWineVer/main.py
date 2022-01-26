import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from svm.accuracy import *
from svm.linear import LinearSVM

start = datetime.datetime.now()
f = pd.read_csv('winequality-red.csv',';')
f.columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
file = f[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]]
file = np.array(file)
y = f[["quality"]]

for i in range(file.shape[0]):
    file[i][file.shape[1]-1] = 0

X_train1, X_test1, y_train1, y_test1 = train_test_split(file, y, test_size=0.3)

y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)

for i in range(y_train1.shape[0]):
    if(y_train1[i] >= 6.5):
        y_train1[i] = 1
    else:
        y_train1[i] = -1
for i in range(y_test1.shape[0]):
    if (y_test1[i] >= 6.5):
        y_test1[i] = 1
    else:
        y_test1[i] = -1

k=0
x_train = np.zeros((6,X_train1.shape[0],2)) #All is 6 parties (2 features in each)
for j in range(6):
    for i in range(X_train1.shape[0]):  #(0~1399, 1400)
        x_train[j][i] = X_train1[i][k:k+2]
    k+=2

k=0
x_test = np.zeros((6,X_test1.shape[0],2))
for j in range(6):
    for i in range(X_test1.shape[0]): #(1400~1598, 199)
        x_test[j][i] = X_test1[i][k:k+2]
    k+=2

Xtrain = x_train[0]   #(1400,12)
for i in range(1, x_train.shape[0]):
    Xtrain = np.hstack([Xtrain, x_train[i]])

Xtest = x_test[0]     #(199,12)
for i in range(1, x_test.shape[0]):
    Xtest = np.hstack([Xtest, x_test[i]])

clf = LinearSVM()
clf.fit(X=x_train, y=y_train1)
#acc_train = accuracy(clf, X=Xtrain, y=y_train1)
acc_test = accuracy(clf, X=Xtest, y=y_test1)
#print("Accuracy (on training): "+str(acc_train))
print("Accuracy (on test): "+str(acc_test))
end = datetime.datetime.now()
print("Processing time： ", end - start)

