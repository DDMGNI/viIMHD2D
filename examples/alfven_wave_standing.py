
import numpy as np

u0 = 1.0
B0 = 1.0
p0 = 0.1

Bx0 = B0
By0 = B0


def magnetic_x(x, y, hx, hy):
    return Bx0

def magnetic_y(x, y, hx, hy):
    return By0 * np.sin(np.pi * x)

def velocity_x(x, y, hx, hy):
    return 0.

def velocity_y(x, y, hx, hy):
    return u0 * np.cos(np.pi * x)

def pressure(x, y, hx, hy):
    return p0
