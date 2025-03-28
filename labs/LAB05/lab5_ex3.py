import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve


def f(x):
    return np.sin(x)
n = 30

fx1 = np.zeros(n+1)
for i in range(n+1):
    fx1[i] = f(i)


x1 = np.zeros(n+1) 
for i in range(n+1):
    x1[i] = 5*i / n

vand = np.vander(x1, increasing=True) 

a1 = np.matmul(np.linalg.inv(vand), f(x1))
print(a1)

p1 = np.zeros(n+1)
Sum1 = 0
for i in range(n+1):
    Sum1 = Sum1 + (a1[i]*(x1[i])**i)
    p1[i] = Sum1
    i += 1

lu, piv = lu_factor(vand)
a2 = lu_solve((lu, piv), f(x1))


print(a2)

p2 = np.zeros(n+1)
Sum2 = 0
for i in range(n+1):
    Sum2 = Sum2 + (a2[i]*(x1[i])**i)
    p2[i] = Sum2

fig = plt.figure(figsize = (12,8))
plt.plot(x1, p1, 'b')
plt.title('p1')  
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("p1")
plt.show()

fig = plt.figure(figsize = (12,8))
plt.plot(x1, p2, 'r')
plt.title('p2')  
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("p2")
plt.show()

fig = plt.figure(figsize = (12,8))
plt.plot(x1, p1, 'b')
plt.plot(x1, p2, 'r')
plt.title('Inverse Multiplication vs LU Factorization')  
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("INVvsLU")
plt.show()
