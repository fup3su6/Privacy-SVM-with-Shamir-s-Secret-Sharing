from cvxopt import matrix, solvers
from .secret_sharing import *
import datetime
solvers.options['show_progress'] = False

def fit_soft(x, y, C):
    #calculate kernal, alpha(solver.pq)
    start = datetime.datetime.now() #2-1. 各個local把data轉成kernel matrix的時間
    featrue = int(x.shape[2])
    num = int(x.shape[1])        #1-d length
    numKernel = int(x.shape[0])  #number of kernels

    average = []
    LK = np.zeros((numKernel, num, featrue))
    LK2 = np.zeros((numKernel, num, num))
    for i in range(numKernel):
        start = datetime.datetime.now()
        LK[i] = y * x[i]                    #K = y * x
        LK2[i] = np.dot(LK[i], LK[i].T)     #K = np.dot(K, K.T)
        end = datetime.datetime.now()
        average.append(end - start)
        #print(end-start)

    s = average[0]
    for i in range(1, len(average)):
        s += average[i]
    s = s / numKernel

    K = np.zeros((num, num))
    for i in range(numKernel):
        K += LK2[i]

    end = datetime.datetime.now()
    t1 = end - start
    # 15/5

    sct = 0
    cS = 0
    k2=0
    g=1
    for i in range(5):  #5 groups (numkernels)
        print("Group "+str(g)+":")
        secret, cloudSum, t2, tt1, s = secret_sharing(LK2[k2:k2+3], 3, num)  #3 parties in a group
        g += 1
        k2+=3
        sct += secret
        cS += cloudSum
        print(" ")
        print("Data to LK + Sum of LK & rMtx on cloud： ", t1 + t2)
        print(" ")

    K = cS - sct
    tt2 = datetime.datetime.now()
    print("Reconstruct secret + Get global kernel： ", tt2 - tt1)
    print(" ")
    #print("P (range: 12~31, Prime = 2^P - 1): ", pp)
    #print("Prime: ", prime)
    #print("coeff field: [0, Prime)")

    print("Sum of 3 level time: ", s + t1 + t2 + tt2 - tt1)

    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    g = np.concatenate((-np.eye(num), np.eye(num)))
    G = matrix(g)
    h_array = np.concatenate((np.zeros(num), C * np.ones(num)))
    h = matrix(h_array)

    A = matrix(y.reshape(1, -1))
    A = matrix(A, (1, num), 'd')

    b = matrix(np.zeros(1))

    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    return alphas

