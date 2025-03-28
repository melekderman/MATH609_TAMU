from math import cos, sin
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def f(t, u):
    return -u


const = np.sin(np.pi/20)
def exact(u0, du0, t):
    return u0 * cos(t) + du0 * sin(t)



def iterate(func, u, v, tmax, n):
    dt = tmax/(n-1)
    t = 0.0

    for i in range(n):
        u,v = func(u,v,t,dt)
        t += dt

    return u



def euler_iter(u, v, t, dt):
    v_new = v + dt * f(t, u)
    u_new = u + dt * v
    return u_new, v_new

def rk_iter(u, v, t, dt):
    k1 = f(t,u)
    k2 = f(t+dt*0.5,u+k1*0.5*dt)
    k3 = f(t+dt*0.5,u+k2*0.5*dt)
    k4 = f(t+dt,u+k3*dt)

    v += dt * (k1+2*k2+2*k3+k4)/6

    # v doesn't explicitly depend on other variables
    k1 = k2 = k3 = k4 = v

    u += dt * (k1+2*k2+2*k3+k4)/6

    return u,v

def Heuns_iter(u, v, t, dt):
    k1 = f(t, u)
    k2 = f(t+dt, u+dt*k1)
    v += dt * (0.5 * (k1+k2))
    u += dt * v
    return u,v

def Heunsiteration(f,a,b,x0,dt):
    t = a
    x = x0
    while t < b+0.5*dt:
        x = Heuns_iter(f,t,x,dt)
        t=t+dt




euler = lambda u, v, tmax, n: iterate(euler_iter, u, v, tmax, n)
runge_kutta = lambda u, v, tmax, n: iterate(rk_iter, u, v, tmax, n)
heuns = lambda u, v, tmax, n: iterate(Heuns_iter,u,v,tmax,n)


def plot_result(u, v, tmax, n):
    dt = tmax/(n-1)
    t = 0.0
    allt = []
    error_euler = []
    error_rk = []
    r_exact = []
    r_euler = []
    r_rk = []
    r_heuns = []

    u0 = u_euler = u_rk = u_heuns = u
    v0 = v_euler = v_rk = v_heuns = v

    for i in range(n):
        u = exact(u0, du0, t)
        u_euler, v_euler = euler_iter(u_euler, v_euler, t, dt)
        u_rk, v_rk = rk_iter(u_rk, v_rk, t, dt)
        u_heuns, v_heuns = Heuns_iter(u_heuns, v_heuns, t, dt)
        allt.append(t)
        error_euler.append(abs(u_euler-u))
        error_rk.append(abs(u_rk-u))
        r_exact.append(u)
        r_euler.append(u_euler)
        r_rk.append(u_rk)
        r_heuns.append(u_heuns)
        t += dt

    #_plot("error_linear_1.png", "Error_linear_1", "time t", "error e", allt, error_euler, error_rk)
    _plot("result_linear_1.png", "Result_linear_1", "time t", "u(t)", allt, r_euler, r_rk, r_heuns, r_exact)


def _plot(out, title, xlabel, ylabel, allt, euler, rk, heuns, exact):
    import matplotlib.pyplot as plt

    plt.title(title)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.plot(allt, euler, 'b-', label="Euler")
    plt.plot(allt, rk, 'r--', label="Runge-Kutta")
    plt.plot(allt, heuns, 'y--', label="Heuns")
    plt.plot(allt, exact, 'g--', label="Exact")



    plt.legend(loc=4)
    plt.grid(True)

    plt.savefig(out, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False)

u0 = 0
du0 = v0 = np.pi/10
tmax = 20
n = 15000

print("t=", tmax)
print("euler =", euler(u0, v0, tmax, n))
print("runge_kutta=", runge_kutta(u0, v0, tmax, n))
print("heuns=", heuns(u0, v0, tmax, n))
print("exact=", exact(u0, du0, tmax))

plot_result(u0, v0, tmax*2, n*2)
