import numpy as np
import datetime
from random import randint

def RandomMatrix(highest, matrixSize):
    return np.random.randint(highest, size=(matrixSize,matrixSize))

def splitMatrix(matrix, shareNum):

    matrixSize = matrix.shape[0]
    sliceMatrix = np.empty((shareNum, matrixSize, matrixSize), dtype=np.int32)

    start = datetime.datetime.now()

    for row in range(matrixSize):
        for column in range(matrixSize):
            pieces = []
            for idx in range(shareNum-1):
                # Number between 1 and matrix[row][column]
                # minus the current total so we don't overshoot
                pieces.append(randint(0,matrix[row][column]-sum(pieces)))
            pieces.append(matrix[row][column]-sum(pieces))
            sliceMatrix[0:shareNum,row,column] = pieces

    end = datetime.datetime.now()
    print("SplitMatrix time： ", end - start)
    print(sliceMatrix.dtype)
    return sliceMatrix
