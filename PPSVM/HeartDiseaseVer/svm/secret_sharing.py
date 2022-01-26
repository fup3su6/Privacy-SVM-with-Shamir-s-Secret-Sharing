from .secretTool2 import *
def secret_sharing(LK, numKernel, num):
    #print("P (range: 12~31, Prime = 2^P - 1): ", pp)
    #print("Prime: ", prime)
    #print("coeff field: [0, Prime)")

    average = []
    rMatrix = np.zeros((numKernel, num, num))      #random number MX
    secretSlice = np.zeros((numKernel, numKernel, num, num, 2), dtype=int) #shares
    for i in range(numKernel):
        start = datetime.datetime.now()  #1. 各個local 產生亂數 + 分割shares 傳給各個local
        rMatrix[i] = RandomMatrix(num)
        secretSlice[i] = getShares(rMatrix[i], numKernel)
        end = datetime.datetime.now()
        average.append(end-start)
        print("Each LK generate random matrix + sharing： ", end - start)

    s=average[0]
    for i in range(1,len(average)):
       s+=average[i]
    s = s/numKernel

    #print("Sum of 1. : ", s)

    time0 = datetime.datetime.now()  #2-2. 各個local 把拿到的shares 加總 傳給cloud的時間
    cloudSum = np.zeros((num, num))
    for i in range(numKernel):
        cloudSum += LK[i] + rMatrix[i]
    time1 = datetime.datetime.now()
    t2 = time1 - time0
    #print("cloudSum: ", t2)
    tt1 = datetime.datetime.now()    #3. Cloud 把 2.收到的東西 還原 secret 最後得到global kernel的時間

    reconstruct_secret = np.zeros((num, num), dtype=int) #sum of rMatrix[]
    for i in range(num):
        for j in range(num):
            reconstruct_secret[i][j] = recover_secret(secretSlice[0:numKernel, 0:numKernel, i, j].tolist())

    return reconstruct_secret, cloudSum, t2, tt1, s