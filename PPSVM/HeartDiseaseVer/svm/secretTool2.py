import functools
import random
from decimal import *
from .secretTool import *

global pp
_RINT = functools.partial(random.SystemRandom().randint, 12)
pp = _RINT(31)
prime = 2 ** pp - 1


def RandomMatrix(matrixSize):
    r = np.zeros((matrixSize, matrixSize), dtype=int)
    for i in range(matrixSize):
        for j in range(matrixSize):
            r[i][j] = _RINT(1000)
    return r
    #return np.random.randint(highest, size=(matrixSize,matrixSize))

def _extended_gcd(a, b):
    """
    Division in integers modulus p means finding the inverse of the
    denominator modulo p and then multiplying the numerator by this
    inverse (Note: inverse of A is B such that A*B % p == 1) this can
    be computed via extended Euclidean algorithm
    http://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Computation
    """
    x = 0
    last_x = 1
    y = 1
    last_y = 0
    while b != 0:
        quot = a // b
        a, b = b, a % b
        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y
    return last_x, last_y

def _divmod(num, den, p):
    """Compute num / den modulo prime p
    To explain what this means, the return value will be such that
    the following is true: den * _divmod(num, den, p) % p == num
    """
    inv, _ = _extended_gcd(den, p)
    return num * inv

def _lagrange_interpolate(x, x_s, y_s, p):
    """
    Find the y-value for the given x, given n (x, y) points;
    k points will define a polynomial of up to kth order.
    """
    k = len(x_s)
    assert k == len(set(x_s)), "points must be distinct"
    def PI(vals):  # upper-case PI -- product of inputs
        accum = 1
        for v in vals:
            accum *= v
        return accum
    nums = []  # avoid inexact division
    dens = []
    for i in range(k):
        others = list(x_s)
        cur = others.pop(i)
        nums.append(PI(x - o for o in others))
        dens.append(PI(cur - o for o in others))
    den = PI(dens)
    num = sum([_divmod(nums[i] * den * y_s[i] % p, dens[i], p)
               for i in range(k)])
    return (_divmod(num, den, p) + p) % p

#拉格朗日重建secret
def recover_secret(shares, p=prime):
    """
    Recover the secret from share points
    (x, y points on the polynomial).
    """
    if len(shares) < 2:
        raise ValueError("need at least two shares")

    tmp = []
    for i in range(len(shares)):
        for j in range(len(shares[i])):
            tmp.append(shares[i][j])

    tmp2 = []
    for i in range(len(shares)):
        tmp2.append(tmp[i])

    for i in range(len(shares)):
        for j in range(len(shares), len(tmp)):
            if tmp2[i][0] == tmp[j][0]:
                tmp2[i][1] += tmp[j][1]

    x_s, y_s = zip(*tmp2)
    return _lagrange_interpolate(0, x_s, y_s, p)

#算y
def polynom( x , coeff ): 
      
    # Evaluates a polynomial in x with coeff being the coefficient list 
    # 計算 x 中的多項式，其中 coeff 是係數列表

    sum = 0
    for i in range(len(coeff)):
        p1 = x ** (len(coeff) - i - 1)
        p1 = p1 * coeff[i]
        sum += p1
    return sum
    #return sum( [ x ** ( len(coeff) - i - 1 ) * coeff[i] for i in range( len(coeff) ) ] )

#算方程式
def coeff( t , secret ): 
    
    # Randomly generate a coefficient array for a polynomial with degree t-1 whose constant = secret
    # 隨機生成一個多項式的係數數組，次數為 t-1，其常數 = secret

    coeff = [_RINT(prime - 1) for i in range(t - 1)]  #field: [0, p)

    coeff.append( secret ) 

    return coeff 

def genX( n ) :
    x = []
    for i in range(n):
        x.append( 3 + i * 2 )
    return x

#產生n個point(x,y)
def generateShares( t , n , secret ) : 
    
    x = genX(n)

    # Split secret using SSS into n shares with threshold t 
    cfs = coeff( t , secret ) 
    shares = []

    for i in range( 0 , n ) :
        shares.append( [ x[i] , (round( polynom ( x[i] , cfs ) , 1 )) % prime ] ) #points are computed mod prime

    return shares 

def generate_n_shares(n, secret):
    t = 3
    shares = generateShares( t , n , secret )
    return shares

def getShares(rMtrix, n):
    tmp_arr = np.zeros((n, rMtrix.shape[0], rMtrix.shape[1], 2), dtype=int)

    #complexity: m*m*numbers of kernel(n)
    for i in range(rMtrix.shape[0]):
        for j in range(rMtrix.shape[1]):
            sharedMatrix = generate_n_shares(n, rMtrix[i][j])
            for k in range(len(sharedMatrix)):
                tmp_arr[k][i][j] = sharedMatrix[k]

    return tmp_arr