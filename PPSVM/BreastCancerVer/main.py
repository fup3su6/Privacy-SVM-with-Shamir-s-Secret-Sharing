import datetime
import random

from sklearn.model_selection import train_test_split
import pandas as pd
from svm.accuracy import *
from svm.linear import LinearSVM

start = datetime.datetime.now()
f = pd.read_csv('dataset_breastcancer.csv')
f.columns=["id", "diagnosis", "radius_2ean", "texture_2ean", "peri2eter_2ean", "area_2ean", "s2oothness_2ean", "co2pactness_2ean", "concavity_2ean", "concave points_2ean", "sy22etry_2ean", "fractal_di2ension_2ean", "radius_se, ", "texture_se, "	, "peri2eter_se, ", "	area_se, ", "	s2oothness_se, ", "	co2pactness_se, ", "	concavity_se, ", "	concave points_se, ", "	sy22etry_se, ", "	fractal_di2ension_se, ", "	radius_worst, ", "	texture_worst, ", "	peri2eter_worst, ", "	area_worst, ", "	s2oothness_worst, ", "	co2pactness_worst, ", "	concavity_worst, ", "	concave points_worst, ", "	sy22etry_worst, ", "	fractal_di2ension_worst"]
file = f[[ "radius_2ean", "texture_2ean", "peri2eter_2ean", "area_2ean", "s2oothness_2ean", "co2pactness_2ean", "concavity_2ean", "concave points_2ean", "sy22etry_2ean", "fractal_di2ension_2ean", "radius_se, ", "texture_se, "	, "peri2eter_se, ", "	area_se, ", "	s2oothness_se, ", "	co2pactness_se, ", "	concavity_se, ", "	concave points_se, ", "	sy22etry_se, ", "	fractal_di2ension_se, ", "	radius_worst, ", "	texture_worst, ", "	peri2eter_worst, ", "	area_worst, ", "	s2oothness_worst, ", "	co2pactness_worst, ", "	concavity_worst, ", "	concave points_worst, ", "	sy22etry_worst, ", "	fractal_di2ension_worst"]]
file = np.array(file)
y = f[["diagnosis"]]

''' 打亂column(Pi)
l1 = random.sample(range(0, 30), 30)  
file = file[:, l1]
print(l1)
'''

X_train1, X_test1, y_train1, y_test1 = train_test_split(file, y, test_size=0.3)

y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)

for i in range(y_train1.shape[0]):
    if(y_train1[i] == 2):
        y_train1[i] = -1
    if(y_train1[i] == 4):
        y_train1[i] = 1
for i in range(y_test1.shape[0]):
    if (y_test1[i] == 2):
        y_test1[i] = -1
    if (y_test1[i] == 4):
        y_test1[i] = 1

k=0
x_train = np.zeros((15,X_train1.shape[0],2))
for j in range(15):
    for i in range(X_train1.shape[0]): #(0~499, 500)
        x_train[j][i] = X_train1[i][k:k+2]
    k+=2

k=0
x_test = np.zeros((15,X_test1.shape[0],2))  #ALL is 15 parties (2 featrues in each)
for j in range(15):
    for i in range(X_test1.shape[0]): #(500~558, 59)
        x_test[j][i] = X_test1[i][k:k+2]
    k+=2

Xtrain = x_train[0]   #(500,30)
for i in range(1, x_train.shape[0]):  #range(1, x_train.shape[0])
    Xtrain = np.hstack([Xtrain, x_train[i]])

Xtest = x_test[0]     #(59,30)
for i in range(1, 15):  #range(1, x_test.shape[0])
    Xtest = np.hstack([Xtest, x_test[i]])

#print(x_train[:3].shape)
print("Dataset: Breast cancer")

clf = LinearSVM()
clf.fit(X=x_train[:15], y=y_train1)

acc_test = accuracy(clf, X=Xtest, y=y_test1)
print("Accuracy (on test): "+str(acc_test))
end = datetime.datetime.now()
#print("All time： ", end - start)
