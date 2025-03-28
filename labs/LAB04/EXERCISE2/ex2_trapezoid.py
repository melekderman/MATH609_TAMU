import numpy as np
import matplotlib.pyplot as plt
def I_trap(n):
    a = 0
    b = 3
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    f = 2*x*np.exp(-1*x**2)
    return (h/2)*(f[0] + 2 * sum(f[1:n-1]) + f[n-1])

I_trap_m = np.zeros(4)
n1 = np.array([5, 10, 15, 20])
for i in range(4):
    I_trap_m[i] = I_trap(n1[i])
print(I_trap_m)

x_axis = np.log10(n1)
print(x_axis)    

I = -1*np.exp(-9)+np.exp(0)
print(I)

y1 = np.zeros(4)
for i in range(4):
    y1[i] = np.log10(np.abs(I_trap_m[i] - I))




alpha_2 = (y1[3]-y1[0])/(np.log10(20)-np.log10(5))
print('alpha2=', alpha_2)

fig = plt.figure(figsize = (10,10))
plt.plot(x_axis, y1, 'b', label = "mid_5")
plt.title('Trapezoid for n=5, 10, 15, 20')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("ex2_trapezoid")
plt.show()