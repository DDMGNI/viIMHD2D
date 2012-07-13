
import numpy as np

u0 = 1.0
B0 = 1.0

Bx0 = B0
By0 = B0


def magnetic_x(x, y, Lx, Ly):
    return Bx0

def magnetic_y(x, y, Lx, Ly):
    return By0 * np.sin(np.pi * x)

def velocity_x(x, y, Lx, Ly):
    return 0.0

def velocity_y(x, y, Lx, Ly):
    return u0 * np.sin(np.pi * x)
