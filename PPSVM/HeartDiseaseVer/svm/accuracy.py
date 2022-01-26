import numpy as np

def accuracy(clf, X, y):
    y_pred = clf.predict(X=X)
    comp = []
    for i in range(y.shape[0]):
        if y_pred[i] == y[i]:
            comp.append(1)
        else:
            comp.append(0)
    return np.mean(comp)