import numpy as np
import matplotlib.pyplot as plt
    
def quad(n):
    a = 0
    b = 3
    alpha = (1/3)**(1/2)
    h = (b - a) / n
    x = np.zeros(n)
    for j in range(n):
        x[j] = a + j * h
    def f(y):
        return 2*y*np.exp(-1*y**2)
    f1 = np.zeros(n)
    sum = 0.0
    for i in range(1,n):
        f1[i] = (f((h * alpha / 2 ) + (x[i] + x[i-1]) / 2) + f((-1 * h * alpha / 2 ) + (x[i-1] + x[i]) / 2))
        sum = sum + f1[i]
    return h/2*sum

n_array = np.array([5, 10, 15, 20])
quad_n = np.zeros(4)
for i in range(4):
    quad_n[i] = quad(n_array[i])
I = -1*np.exp(-9)+np.exp(0)
x_axis = np.log10(n_array)
y1 = np.zeros(4)
for i in range(4):
    y1[i] = np.log10(np.abs(quad_n[i] - I))

alpha_4 = (y1[2]-y1[0])/(np.log10(5)-np.log10(10))
print('alpha4=', alpha_4)

fig = plt.figure(figsize = (12,8))
plt.plot(x_axis, y1, 'b')
plt.title('Quadrature for n=5, 10, 15, 20')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("ex2_quadrature")
plt.show()

