import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 2*x*np.exp(-1*x**2)

def midpoint(a, b, n):
    """
    midpoint rule numerical integral implementation
    a: interval start
    b: interval end
    n: number of steps
    return: numerical integral evaluation
    """

    # initialize result variable
    res = 0

    # calculate number of steps
    h = (b - a) / n

    # starting midpoint
    x = a + (h / 2)

    # evaluate f(x) at subsequent midpoints
    for _ in range(n):
        res += f(x)
        x += h

    # multiply final result by step size
    return h * res


I = -1*np.exp(-9)+np.exp(0)
print(I)

mid_n = np.zeros(4)
n_array = np.array([5, 10, 15, 20])
for i in range(4):
    mid_n[i] = midpoint(0, 3, n_array[i])

print(mid_n)
x_axis = np.log10(n_array)
#for i in range(4):
#    x_axis[i] = np.log10(n[i])
print(x_axis)    
y1 = np.zeros(4)
for i in range(4):
    y1[i] = np.log10(np.abs(mid_n[i] - I))

print(y1)


alpha_1 = (y1[2]-y1[0])/(np.log10(5)-np.log10(15))
print('alpha1=', alpha_1)


fig = plt.figure(figsize = (12,8))
plt.plot(x_axis, y1, 'b', label = "mid_5")
plt.title('Midpoint Rule for n=5, 10, 15, 20')  
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("ex2_midpoint")
plt.show()