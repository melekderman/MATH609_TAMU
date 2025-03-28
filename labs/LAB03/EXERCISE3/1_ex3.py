#FOURIER TRANSFORM AND NOISY SIGNALS

import numpy as np
import matplotlib.pyplot as plt

n = 500
R = 0
f = np.zeros(n)
x = np.zeros(n)

for i in range(0,n):
    x[i] = 2*np.pi*i/n
    i += 1
print(x)

for i in range(0,n):
    f[i] = np.sin(x[i]) + np.cos(3*x[i]) - np.sin(7*x[i])
    i += 1
print(f)

y = np.zeros(n)
N = np.zeros(n)
for i in range(0,n):
    N[i] = np.random.normal(loc=0.0, scale= 2*np.pi)
for i in range(0,n):
    y[i] = f[i] + R * N[i]

y1 = np.fft.fft(y)

m = 10
for i in range(0,n):
    if i>m and i<n-m:
        y1[i] = 0
    i += 1
yi = np.fft.ifft(y1)



plt.figure()
plt.plot(x, yi)
plt.plot(x, f)
plt.show()

print(N)


