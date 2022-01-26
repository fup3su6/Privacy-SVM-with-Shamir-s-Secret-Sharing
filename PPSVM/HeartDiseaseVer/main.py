import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from svm.accuracy import *
from svm.linear import LinearSVM
import random

start = datetime.datetime.now()
f = pd.read_csv('heart.csv')
f.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target", "n"]
file = f[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target", "n"]]
file = np.array(file)
y = f[["target"]]

for i in range(file.shape[0]):
    file[i][file.shape[1] - 2] = 0
    file[i][file.shape[1] - 1] = 0

l1 = random.sample(range(0, 15), 15)  #打亂column(Pi)
file = file[:, l1]


X_train1, X_test1, y_train1, y_test1 = train_test_split(file, y, test_size=0.3)

y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)

for i in range(y_train1.shape[0]):
    if(y_train1[i] == 0):
        y_train1[i] = -1
for i in range(y_test1.shape[0]):
    if (y_test1[i] == 0):
        y_test1[i] = -1

k=0
x_train = np.zeros((15,X_train1.shape[0],1)) #All is 15 parties (1 feature in each)
for j in range(15):
    for i in range(X_train1.shape[0]): #(0~259, 260)
        x_train[j][i] = X_train1[i][k:k+1]
    k+=1

k=0
x_test = np.zeros((15,X_test1.shape[0],1))
for j in range(15):
    for i in range(X_test1.shape[0]): #(260~302, 43)
        x_test[j][i] = X_test1[i][k:k+1]
    k+=1

Xtrain = x_train[0]   #(260,15)
for i in range(1, x_train.shape[0]):
    Xtrain = np.hstack([Xtrain, x_train[i]])

Xtest = x_test[0]     #(43,15)
for i in range(1, x_test.shape[0]):
    Xtest = np.hstack([Xtest, x_test[i]])

clf = LinearSVM()
clf.fit(X=x_train, y=y_train1)
#acc_train = accuracy(clf, X=Xtrain, y=y_train1)
acc_test = accuracy(clf, X=Xtest, y=y_test1)
#print("Accuracy (on training): "+str(acc_train))
print("Accuracy (on test): "+str(acc_test))
end = datetime.datetime.now()
print("All time： ", end - start)

