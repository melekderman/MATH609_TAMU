import numpy as np

f = lambda x: (x**4 - 2)**5
f_prime = lambda x: 20*x**3*(x**4-2)**4

def bisection(f, a, b, niter, tol): 

    m = (a + b)/2
    if np.abs(f(m)) < tol:
        print("Number of Iteration for Bisection Method: ", niter)
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        niter_l = niter + 1
        return bisection(f, m, b, niter_l, tol)

    elif np.sign(f(b)) == np.sign(f(m)):
        niter_l = niter + 1
        return bisection(f, a, m, niter_l, tol)

def newton(f, df, x0, niter, tol):
    if abs(f(x0)) < tol:
        print("Number of Iteration for Newton Method: ", niter)
        return x0
    else:
        niter_l = niter + 1
        return newton(f, df, x0 - f(x0)/df(x0), niter_l, tol)



Bisection_Method = bisection(f, 1, 2, 1, 10**-7)
epsilon_B = np.abs(1.189207115 - Bisection_Method) / 1.189207115
print("Bisection Method Estimate:", Bisection_Method)
print("Accuracy:",epsilon_B )

Newton_Method = newton(f, f_prime, 1, 1, 10**-7)
epsilon_NR = np.abs(1.189207115 - Newton_Method) / 1.189207115
print("Newton Method Estimate:", Newton_Method)
print("Accuracy:",epsilon_NR )