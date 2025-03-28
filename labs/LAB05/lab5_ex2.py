import numpy as np
import scipy.linalg

n = 5

while n>4 and n<24:
    x = np.zeros(n+1)
    for i in range(n+1):
        x[i] = 5*(i) / n
    vand = np.vander(x, increasing=True)

    det = 1
    for i in range(n+1):
        for j in range(i+1, n+1):
            det = det * (x[j] - x[i])

    P, L, U = scipy.linalg.lu(vand) 
    det_vand = np.linalg.det(vand)
    det_V_PLU = np.linalg.det(P) * np.linalg.det(L) * np.linalg.det(U)
    
    Relative_Error = abs(det - det_V_PLU) / det
    print(f"Determinant V[{n}]                              =", det)
    print(f"Library Determinant V[{n}]                      =", det_vand)  
    print(f"LU Factorization Determinant V[{n}]             =", det_V_PLU) 
    print(f"Relative Error V[{n}]                           =", Relative_Error )
    n += 1