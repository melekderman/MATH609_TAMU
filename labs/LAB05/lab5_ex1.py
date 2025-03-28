import pprint

import numpy as np
import scipy.linalg  # SciPy Linear Algebra Library
from scipy.linalg import lu_factor, lu_solve

A = np.array([[1, 4, 6], [2, 6, 17], [1, 8, 19]])
b = np.array([11, 29, 28])
P, L, U = scipy.linalg.lu(A)

print("A:")
pprint.pprint(A)

print("P:")
pprint.pprint(P)

print("L:")
pprint.pprint(L)

print("U:")
pprint.pprint(U)

print("LU:")
pprint.pprint(np.matmul(np.matmul(P, L), U))

lu, piv = lu_factor(A)
x = lu_solve((lu, piv), b)
print("x=", x)