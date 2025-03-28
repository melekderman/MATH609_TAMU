import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange



def f(x):
    return np.sin(x)

def xk_a(n):
    ka = np.zeros(2*n+1)
    for i in range(2*n+1):
        ka[i] = -n+i
    xa = np.zeros(2*n+1)
    for i in range(2*n+1):
        xa[i] = 5*ka[i]/n
    return xa

def xk_b(n):
    kb = np.zeros(2*n+2)
    for i in range(2*n+2):
        kb[i] = i
    xb = np.zeros(2*n+2)
    for i in range(2*n+2):
        xb[i] = 5*np.cos(kb[i]*np.pi/(2*n+1))
    return xb


def xk_c(n):
    xc = np.zeros(2*n+1)
    for i in range(2*n+1):
        xc[i] = np.random.uniform(-5, 5)
    return xc


def divided_diff(x, y):

    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def newton_poly(coef, x_data, x):

    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p

#perturbed

Yk5  = np.random.normal(0,0.1, size=(11))
Ykb5  = np.random.normal(0,0.1,size=(12))
Yk10 = np.random.normal(0,0.1,size=(21))
Ykb10  = np.random.normal(0,0.1,size=(22))
Yk20 = np.random.normal(0,0.01,size=(41))
Ykb20  = np.random.normal(0,0.1,size=(42))

xa5  = xk_a(5)
xa10 = xk_a(10)
xa20 = xk_a(20)


xb5  = xk_b(5)
xb10 = xk_b(10)
xb20 = xk_b(20)


xc5  = np.sort(xk_c(5))
xc10 = np.sort(xk_c(10))
xc20 = np.sort(xk_c(20))


ya5   = f(xa5) + Yk5 * 0.01
ya10  = f(xa10) + Yk10 * 0.01
ya20  = f(xa20) + Yk20 * 0.01

yb5   = f(xb5) + Ykb5 * 0.01
yb10  = f(xb10) + Ykb10 * 0.01
yb20  = f(xb20) + Ykb20 * 0.01


yc5   = f(xc5) + Yk5 * 0.01
yc10  = f(xc10) + Yk10 * 0.01
yc20  = f(xc20) + Yk20 * 0.01

x_new = np.arange(-5, 5, .1)

coeffa5 = divided_diff(xa5, ya5)[0, :]
coeffa10 = divided_diff(xa10, ya10)[0, :]
coeffa20 = divided_diff(xa20, ya20)[0, :]

coeffb5 = divided_diff(xb5, yb5)[0, :]
coeffb10 = divided_diff(xb10, yb10)[0, :]
coeffb20 = divided_diff(xb20, yb20)[0, :]

coeffc5 = divided_diff(xc5, yc5)[0, :]
coeffc10 = divided_diff(xc10, yc10)[0, :]
coeffc20 = divided_diff(xc20, yc20)[0, :]

ya5_new = newton_poly(coeffa5, xa5, x_new)
ya10_new = newton_poly(coeffa10, xa10, x_new)
ya20_new = newton_poly(coeffa20, xa20, x_new)

yb5_new = newton_poly(coeffb5, xb5, x_new)
yb10_new = newton_poly(coeffb10, xb10, x_new)
yb20_new = newton_poly(coeffb20, xb20, x_new)

yc5_new = newton_poly(coeffc5, xc5, x_new)
yc10_new = newton_poly(coeffc10, xc10, x_new)
yc20_new = newton_poly(coeffc20, xc20, x_new)

laga5 = lagrange(xa5,ya5)
laga10 = lagrange(xa10,ya10)
laga20 = lagrange(xa20,ya20)

lagb5 = lagrange(xb5,yb5)
lagb10 = lagrange(xb10,yb10)
lagb20 = lagrange(xb20,yb20)

lagc5 = lagrange(xc5,yc5)
lagc10 = lagrange(xc10,yc10)
lagc20 = lagrange(xc20,yc20)


#plot 1-a
plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.plot(xa5, ya5, 'bo')
plt.plot(x_new, ya5_new, 'r')
plt.plot(x_new, laga5(x_new), '1')
plt.savefig('ex2a perturbed spf _ n = 5')


plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xa10, ya10, 'bo')
plt.plot(x_new, ya10_new, 'r')
plt.plot(x_new, laga10(x_new), '1')
plt.savefig('ex2a perturbed spf _ n = 10')


plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xa20, ya20, 'bo')
plt.plot(x_new, ya20_new, 'r')
plt.plot(x_new, laga20(x_new), 'g')
plt.savefig('ex2a perturbed spf _ n = 20')


#plot 1-b
plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xb5, yb5, 'bo')
plt.plot(x_new, yb5_new, 'r')
plt.plot(x_new, lagb5(x_new),'1')
plt.savefig('ex2b perturbed spf _ n = 5')


plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xb10, yb10, 'bo')
plt.plot(x_new, yb10_new, 'r')
plt.plot(x_new, lagb10(x_new),'1')
plt.savefig('ex2b perturbed spf _ n = 10')


plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xb20, yb20, 'bo')
plt.plot(x_new, yb20_new, 'r')
plt.plot(x_new, lagb20(x_new),'g')
plt.savefig('ex2b perturbed spf _ n = 20')


#plot 1-c
plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xc5, yc5, 'bo')
plt.plot(x_new, yc5_new, 'r')
plt.plot(x_new, lagc5(x_new),'1')
plt.savefig('ex2c perturbed spf _ n = 5')


plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xc10, yc10, 'bo')
plt.plot(x_new, yc10_new, 'r')
plt.plot(x_new, lagc10(x_new), '1')
plt.savefig('ex2c perturbed spf _ n = 10')


plt.figure(figsize = (12, 8))
plt.xlim(-5,5)
plt.ylim(-1.1,1.1)
plt.plot(xc20, yc20, 'bo')
plt.plot(x_new, yc20_new, 'r')
plt.plot(x_new, lagc20(x_new), 'g')
plt.savefig('ex2c3 perturbed spf _ n = 20')