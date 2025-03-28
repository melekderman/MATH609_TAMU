import numpy as np
from scipy.linalg import lu_factor, lu_solve

#LU SOLVER:
def LU_Solver(A,b):
    lu, piv = lu_factor(A)
    a = lu_solve((lu, piv), b)
    return a


def jacobi(A, x, b, tolerance, max_iterations):

    niter = 0
    D = - np.diag(np.diagonal(A))
    for k in range(max_iterations):
        niter += 1   
        x_old  = x.copy()
        x = x + np.matmul(np.linalg.inv(D), (b - np.matmul(x, A)))
        Lnorm1 = np.linalg.norm(x - x_old, ord=1) / np.linalg.norm(x, ord=1)
        if Lnorm1 < tolerance or Lnorm1 > 500:
            break
    print("JACOBI Number of Iteration:", niter)
    return x 

def gauss_seidel(A, x, b, tolerance, max_iterations):
    niter = 0
    L = np.tril(A)
    D = np.diag(np.diagonal(A))
    for k in range(max_iterations):
        niter += 1   
        x_old  = x.copy()
        x = x + np.matmul(np.linalg.inv(np.add(D, L)), (b - np.matmul(x, A)))
        Lnorm1 = np.linalg.norm(x - x_old, ord=1) / np.linalg.norm(x, ord=1)
        if Lnorm1 < tolerance or Lnorm1 > 500:
            break
    print("GAUSS-SEIDEL Number of Iteration:", niter)
    return x 

def GD(A, x, b, s, tolerance, max_iterations):
    niter = 0
    r = b - np.matmul(x, A)
    r = np.reshape(r, (1,s))
    p  = r.copy()
    for k in range(max_iterations):
        niter += 1   
        x_old  = x.copy()
        r_old  = r.copy()
        p_old  = p.copy()
        alfa = np.matmul(np.transpose(r), r) / np.matmul(np.matmul(np.transpose(p), p), A)
        x = x + np.matmul(p,alfa)
        r = r - np.matmul(p, np.matmul(alfa,A))
        Lnorm1 = np.linalg.norm(x - x_old, ord=1) / np.linalg.norm(x, ord=1)
        if Lnorm1 < tolerance or Lnorm1 > 500:
            break
        beta = np.matmul(np.transpose(r),r)/np.matmul(np.transpose(r_old),r_old)
        p = r + np.matmul(p,beta)

    print("GD Number of Iteration:", niter)
    return x 



X1 = np.random.uniform(0, 1, size = (15,500))
XXt1 = np.matmul(X1, np.transpose(X1))
a1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, -4, -4, -4, 6, -6])
N =  np.random.standard_normal(size=500)
y1 = np.add(np.matmul(a1, X1), N)
yXt1 = np.matmul(y1, np.transpose(X1))
s1 = 15



X6 = np.random.uniform(0, 1, size = (50,500))
XXt6 = np.matmul(X6, np.transpose(X6))
a6 = np.random.normal(1,3, size=50)
y6 = np.add(np.matmul(a6, X6), N)
yXt6 = np.matmul(y6, np.transpose(X6))
s6 = 50

print("L1 NORM:")

print("d = 15")
print("LU 15 - tol = 0.01:",LU_Solver(XXt1,yXt1))
print("Jacobi 15 - tol = 0.01:", jacobi(XXt1, a1, yXt1, 0.01, 200))
print("Gauss-Seidel 15 - tol = 0.01:", gauss_seidel(XXt1, a1, yXt1, 0.01, 200))
print("GD 15: - tol = 0.01", GD(XXt1, a1, yXt1, s1, 0.01, 200))

print("Jacobi 15 - tol = 0.0001:", jacobi(XXt1, a1, yXt1, 0.0001, 200))
print("Gauss-Seidel 15 - tol = 0.0001:", gauss_seidel(XXt1, a1, yXt1, 0.0001, 200))
print("GD 15: - tol = 0.0001", GD(XXt1, a1, yXt1, s1, 0.0001, 200))

print("d = 50")
print("LU 50: ",LU_Solver(XXt6,yXt6))
print("Jacobi 50: - tol = 0.01", jacobi(XXt6, a6, yXt6, 0.01, 200))
print("Gauss-Seidel 50: - tol = 0.01", gauss_seidel(XXt6, a6, yXt6, 0.01, 200))
print("GD 50: - tol = 0.01", GD(XXt6, a6, yXt6, s6, 0.01, 200))

print("Jacobi 50: - tol = 0.0001", jacobi(XXt6, a6, yXt6, 0.0001, 200))
print("Gauss-Seidel 50: - tol = 0.0001", gauss_seidel(XXt6, a6, yXt6, 0.0001, 200))
print("GD 50: - tol = 0.0001", GD(XXt6, a6, yXt6, s6, 0.0001, 200))