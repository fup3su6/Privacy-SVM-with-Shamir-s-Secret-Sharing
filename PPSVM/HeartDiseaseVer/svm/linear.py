from .solver import *
import math

class LinearSVM(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None
    def fit(self, X, y):  #soft=True
        C = 1.0
        alphas = fit_soft(X, y, C)

        XX = X[0]    #original x_train(500, 30)
        for i in range(1, X.shape[0]):
            XX = np.hstack([XX, X[i]])

        # get weights
        w = np.sum(alphas * y * XX, axis=0)
        # get b
        b_vector = y - np.dot(XX, w)
        b = b_vector.sum() / b_vector.size

        # normalize
        norm = np.linalg.norm(w)
        w, b = w / norm, b / norm

        # Store values
        self.w = w
        self.b = b

        aa = alphas
        aa = np.power(aa, 2)
        lr = 0
        for i in range(aa.shape[0]):
            lr += aa[i]
        lr = math.sqrt(lr)
        lr = lr/alphas.shape[0]
        print("Learning rate: "+str(lr))

        #print(w)
        #print(b)
    def predict(self, X):
        y = np.sign(np.dot(self.w, X.T) + self.b * np.ones(X.shape[0]))
        return y