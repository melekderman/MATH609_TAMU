import numpy as np
import matplotlib.pyplot as plt
def I_simp(n):
    a = 0
    b = np.pi
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    f = 2*x*np.exp(-1*x**2)
    return (h/3) * (f[0] + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])

I_simp_m = np.zeros(4)
n = np.array([5, 10, 15, 20])
for i in range(4):
    I_simp_m[i] = I_simp(n[i])
print(I_simp_m)

x_axis = np.log10(n)
print(x_axis)    

I = -1*np.exp(-9)+np.exp(0)

y1 = np.zeros(4)
for i in range(4):
    y1[i] = np.log10(np.abs(I_simp_m[i] - I))

alpha_3 = (y1[2] - y1[0])/(np.log10(5)-np.log10(15))
print(alpha_3)

fig = plt.figure(figsize = (12,8))
plt.plot(x_axis, y1, 'b', label = "mid_5")
plt.title("Simpson's Rule for n=5, 10, 15, 20")
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("ex2_simpson")
plt.show()


