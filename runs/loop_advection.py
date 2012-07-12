
import numpy as np

A0 = 1E-3
R0 = 0.3
p0 = 1.0
u0 = np.sqrt(5.)
th = np.arctan(0.5)


def magnetic_x(x, y, Lx, Ly):
    if x == 0 and y == 0:
        return 0.
    else:
        return - A0 * y / np.sqrt(x**2 + y**2)

def magnetic_y(x, y, Lx, Ly):
    if x == 0 and y == 0:
        return 0.
    else:
        return + A0 * x / np.sqrt(x**2 + y**2)

def velocity_x(x, y, Lx, Ly):
    return u0 * np.cos(th)

def velocity_y(x, y, Lx, Ly):
    return u0 * np.sin(th)

def pressure(x, y, Lx, Ly):
    return p0
