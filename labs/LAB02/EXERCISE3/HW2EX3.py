import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def x1(n):
    return np.arange(0,n,1)

def f1(x):
    return 1/(1+x**2)

def g1(x):
    return (f1(x+1)+f1(x))/2


x0_5  = np.arange(0,6,1)
x0_10 = np.arange(0,11,1)
x0_11 = np.arange(0,12,1)
x0_20 = np.arange(0,21,1)
x0_30 = np.arange(0,31,1)
x0_50 = np.arange(0,51,1)
x0_150 = np.arange(0,151,1)
x0_300 = np.arange(0,301,1)

y0_5  = f1(x0_5)
y0_10 = f1(x0_10)
y0_11 = f1(x0_11)
y0_20 = f1(x0_20)
y0_30 = f1(x0_30)
y0_50 = f1(x0_50)
y0_150 = f1(x0_150)
y0_300 = f1(x0_300)



y1_5  = g1(x0_5)
y1_10 = g1(x0_10)
y1_20 = g1(x0_20)
y1_30 = g1(x0_30)
y1_50 = g1(x0_50)
y1_150 = g1(x0_150)
y1_300 = g1(x0_300)

y2_5  = scipy.interpolate.interp1d(x0_5, y0_5, kind='linear')
y2_10 = scipy.interpolate.interp1d(x0_10, y0_10, kind='linear')
y2_20 = scipy.interpolate.interp1d(x0_20, y0_20, kind='linear')
y2_30 = scipy.interpolate.interp1d(x0_30, y0_30, kind='linear')
y2_50 = scipy.interpolate.interp1d(x0_50, y0_50, kind='linear')
y2_150 = scipy.interpolate.interp1d(x0_150, y0_150, kind='linear')
y2_300 = scipy.interpolate.interp1d(x0_300, y0_300, kind='linear')

y3_5  = scipy.interpolate.CubicSpline(x0_5, y0_5)
y3_10 = scipy.interpolate.CubicSpline(x0_10, y0_10)
y3_11  = scipy.interpolate.CubicSpline(x0_11, y0_11)
y3_20 = scipy.interpolate.CubicSpline(x0_20, y0_20)
y3_30 = scipy.interpolate.CubicSpline(x0_30, y0_30)
y3_50 = scipy.interpolate.CubicSpline(x0_50, y0_50)
y3_150 = scipy.interpolate.CubicSpline(x0_150, y0_150)
y3_300 = scipy.interpolate.CubicSpline(x0_300, y0_300)

x5new = np.arange(0, 5, 0.1)
x10new = np.arange(0, 10, 0.1)
x11new = np.arange(0, 11, 0.1)
x20new = np.arange(0, 20, 0.1)
x30new = np.arange(0, 30, 0.1)
x50new = np.arange(0, 50, 0.1)
x150new = np.arange(0, 150, 0.1)
x300new = np.arange(0, 300, 0.1)

g2_5 = y2_5(x5new)
g2_10 = y2_10(x10new)
g2_20 = y2_20(x20new)
g2_30 = y2_30(x30new)
g2_50 = y2_50(x50new)
g2_150 = y2_150(x150new)
g2_300 = y2_300(x300new)

g3_5 = y3_5(x5new)
g3_10 = y3_10(x10new)
g3_11 = y3_11(x11new)
g3_20 = y3_20(x20new)
g3_30 = y3_30(x30new)
g3_50 = y3_50(x50new)
g3_150 = y3_150(x150new)
g3_300 = y3_300(x300new)

def I1(N):
    f = y0_300
    g = y1_300
    sum1 = 0.0
    for j in range(N):
        sum1 = sum1 + np.abs(f[j]-g[j])*(-5+10*(j+1)/(N+1))
        j += 1
    return sum1/N

def I2_150(N):
    f = y0_150
    g = g2_150
    sum1 = 0.0
    for j in range(N):
        sum1 = sum1 + np.abs(f[j]-g[j])*(-5+10*(j+1)/(N+1))
        j += 1
    return sum1/N

def I2_300(N):
    f = y0_300
    g = g2_300
    sum1 = 0.0
    for j in range(N):
        sum1 = sum1 + np.abs(f[j]-g[j])*(-5+10*(j+1)/(N+1))
        j += 1
    return sum1/N

def I3_150(N):
    f = y0_150
    g = g3_150
    sum1 = 0.0
    for j in range(N):
        sum1 = sum1 + np.abs(f[j]-g[j])*(-5+10*(j+1)/(N+1))
        j += 1
    return sum1/N

def I3_300(N):
    f = y0_300
    g = g3_300
    sum1 = 0.0
    for j in range(N):
        sum1 = sum1 + np.abs(f[j]-g[j])*(-5+10*(j+1)/(N+1))
        j += 1
    return sum1/N
I1_300a = np.zeros(300)
I2_300a = np.zeros(300)
I3_300a = np.zeros(300)




#EXERCISE 3-1
plt.figure()
plt.plot(x0_11, y0_11, x11new, g3_11)
plt.savefig('ex3_1 n=11')
plt.show()

#EXERCISE 3-2
plt.figure()
plt.plot(x0_5, y0_5, x0_5, y1_5)
plt.savefig('ex3_2a n=5')
plt.figure()
plt.plot(x0_10, y0_10, x0_10, y1_10)
plt.savefig('ex3_2a n=10')
plt.figure()
plt.plot(x0_20, y0_20, x0_20, y1_20)
plt.savefig('ex3_2a n=20')
plt.figure()
plt.plot(x0_30, y0_30, x0_30, y1_30)
plt.savefig('ex3_2a n=30')
plt.figure()
plt.plot(x0_50, y0_50, x0_50, y1_50)
plt.savefig('ex3_2a n=50')
plt.show()


plt.figure()
plt.plot(x0_5, y0_5, x5new, g2_5)
plt.savefig('ex3_2b n=5')
plt.figure()
plt.plot(x0_10, y0_10, x10new, g2_10)
plt.savefig('ex3_2b n=10')
plt.figure()
plt.plot(x0_20, y0_20, x20new, g2_20)
plt.savefig('ex3_2b n=20')
plt.figure()
plt.plot(x0_30, y0_30, x30new, g2_30)
plt.savefig('ex3_2b n=30')
plt.figure()
plt.plot(x0_50, y0_50, x50new, g2_50)
plt.savefig('ex3_2b n=50')
plt.show()

plt.figure()
plt.plot(x0_5, y0_5, x5new, g3_5)
plt.savefig('ex3_2c n=5')
plt.figure()
plt.plot(x0_10, y0_10, x10new, g3_10)
plt.savefig('ex3_2c n=10')
plt.figure()
plt.plot(x0_20, y0_20, x20new, g3_20)
plt.savefig('ex3_2c n=20')
plt.figure()
plt.plot(x0_30, y0_30, x30new, g3_30)
plt.savefig('ex3_2c n=30')
plt.figure()
plt.plot(x0_50, y0_50, x50new, g3_50)
plt.savefig('ex3_2c n=50')
plt.show()

#EXERCISE 3.3

print("Exercise 3.3 _ In for data set a", I1(300))

print("Exercise 3.3 _ In for data set b", I2_300(300))

print("Exercise 3.3 _ In for data set c", I3_300(300))

#EXERCISE 3.4
xlog_300 = np.arange(0,300,1)
plt.figure()
plt.plot(np.log(xlog_300), np.log(I1_300a))
plt.figure()
plt.plot(np.log(xlog_300), np.log(I2_300a))
plt.figure()
plt.plot(np.log(xlog_300), np.log(I3_300a))
plt.show()

