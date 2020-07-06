import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import solve_triangular
import math
import time

MAX_ITER = 10000

def Jacobi(A,b, x, eps):
    D = np.diag(A)
    D_1 = np.diagflat([1/x for x in (D)])   # D^-1
    D_1b = np.dot(D_1,b)                    # D^-1 * b
    LU = A - np.diagflat(D)                 # L+U
    i = 0
    r = np.dot(A,x) - b
    residuum = np.linalg.norm(r)
    xi = x
    startTime = time.time()
    while residuum > eps:
        #tmp = -1 * np.dot(LU,xi)
        #tmp2 = np.dot(tmp,D_1)
        #xi = tmp2 + D_1b
        xi = np.dot((-1*np.dot(LU,xi)) ,D_1) + D_1b     # xi+1 = D^-1 * ( (L+U) * xi) ) + D^1*b 
        r = np.dot(A, xi) - b
        residuum = np.linalg.norm(r)
        i += 1
        if i > MAX_ITER or np.isnan(residuum):
            print("Metoda Jacobiego nie zbiega się\n")
            return None
    endTime = time.time()
    durationTime = endTime - startTime
    print("----Metoda Jacobiego----")
    print("Czas wykonania: " + str(durationTime) + "s.")
    print("Liczba iteracji: " + str(i))
    print("Residuum: " + str(residuum) + "\n\n")
    return durationTime



def Gauss_Seidel(A,b,x,eps):
    i = 0
    r = np.dot(A,x) - b
    residuum = np.linalg.norm(r)
    xi = x
    L = np.tril(A) # Macierz trojkatna dolna + diagonala
    U = A - L
    DLb =  solve_triangular(L,b, lower=True)
    startTime = time.time()
    while residuum > eps:
        xi = solve_triangular(-1*L,np.dot(U,xi), lower=True) + DLb
        r = np.dot(A, xi) - b
        residuum = np.linalg.norm(r)
        i += 1
        if i > MAX_ITER or np.isnan(residuum):
            print("Metoda Gaussa-Seidla nie zbiega się\n")
            return None
    endTime = time.time()
    durationTime = endTime - startTime
    print("----Metoda Gaussa-Seidla----")
    print("Czas wykonania: " + str(durationTime) + "s.")
    print("Liczba iteracji: " + str(i))
    print("Residuum: " + str(residuum) + "\n\n")
    return durationTime


def LUFactorization(A,b,x):
    startTime = time.time()
    m = len(b)
    L = np.eye(m)
    U = np.copy(A)
    r = np.dot(A,x) - b
    residuum = np.linalg.norm(r)
    x1 = x
    for k in range (0,m):
        for j in range(k+1,m):
            L[j,k] = float(U[j,k]) / float(U[k,k])
            U[j, k:m] = U[j, k:m] - (L[j, k] * U[k, k:m])
    tmp = solve_triangular(L,b,lower=True)
    x1 = solve_triangular(U,tmp, lower=False)
    r = np.dot(A,x1) - b
    residuum = np.linalg.norm(r)
    endTime = time.time()
    print("----Metoda LU----")
    print("Czas wykonania: " + str(endTime - startTime) + "s.")
    print("Residuum: " + str(residuum) + "\n\n")
    return endTime - startTime


a1 = 6      # 5 + 1
a2 = -1
a3 = -1
N = 966    
eps = 10e-09

# Tworzenie macierzy A do Zadania A.
A = diags([a3,a2,a1,a2,a3], [-2,-1, 0, 1, 2], shape=(N,N)).toarray()

# Tworzenie wektora b do Zadania A i C.
b=[]
for i in range(0, N):
    b.append(math.sin(i*(3))) # 2 + 1

x = np.zeros(N)
x2 = np.zeros(N)
x3 = np.zeros(N)
x4 = np.zeros(N)
x5 = np.zeros(N)

Jacobi(A,b,x,eps)
Gauss_Seidel(A,b,x2,eps)


A_ = diags([a3,a2,3,a2,a3], [-2,-1, 0, 1, 2], shape=(N,N)).toarray()

Jacobi(A_,b,x3,eps)
Gauss_Seidel(A_,b,x4,eps)

## Zadanie D
LUFactorization(A_,b,x)


# Zadanie E
N1 = [100,500,1000,1500, 2000,3000,4000]
An = []
xn = []
bn = []
for i in range(0,len(N1)):
    An.append(diags([a3,a2,a1,a2,a3], [-2,-1, 0, 1, 2], shape=(N1[i],N1[i])).toarray())
    xn.append(np.zeros(N1[i]))
    bn.append([])
    for j in range(0,N1[i]):
        bn[i].append(math.sin(j*(3)))

jacobiTimes = []
gaussTimes = []
LUTimes = []

for i in range(0,len(N1)):
    jacobiTimes.append(Jacobi(An[i], bn[i], xn[i], eps))
    gaussTimes.append(Gauss_Seidel(An[i], bn[i], xn[i], eps))
    LUTimes.append(LUFactorization(An[i], bn[i], xn[i]))

# Prezentacja wykresów

plt.figure(num="Wojciech Neiewiadomski 172166")
plt.xlabel('liczba niewiadomych N') 
plt.ylabel('czas wykonania [s]')
wykres = plt.plot(N1, jacobiTimes, 'r-', N1, gaussTimes, 'b-', N1, LUTimes, 'g-')
plt.grid(color='gray', linestyle='-', linewidth=0.07)
plt.legend(('Metoda Jacobiego','Metoda Gaussa-Seidla', 'Faktoryzacja LU'),loc = 'upper right')
plt.show()
