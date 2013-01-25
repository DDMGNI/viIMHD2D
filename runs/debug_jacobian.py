
import numpy as np

u0 = 1.0
B0 = 1.0
p0 = 0.1

Bx0 = B0
By0 = B0


def magnetic_x(x, y, Lx, Ly):
    return Bx0 * np.cos(np.pi * x)

def magnetic_y(x, y, Lx, Ly):
    return By0 * np.sin(np.pi * x)

def velocity_x(x, y, Lx, Ly):
    return u0 * np.sin(np.pi * y)
#    return 1.

def velocity_y(x, y, Lx, Ly):
    return u0 * np.cos(np.pi * x)
#    return 1.

def pressure(x, y, Lx, Ly):
    return p0
