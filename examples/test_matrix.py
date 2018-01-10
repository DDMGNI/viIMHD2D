
import numpy as np

u0 = 0.1
B0 = 1.0
p0 = 0.1

#u0 = 1.0
#B0 = 0.0
#p0 = 1.0

x1 = 0.5
x2 = 1.5


def magnetic_x(x, y, Lx, Ly):
    return 0.0

def magnetic_y(x, y, Lx, Ly):
    if x1 <= x and x <= x2:
        return - B0
    else:
        return + B0
    
def velocity_x(x, y, Lx, Ly):
    return u0 * np.sin(np.pi * y)

def velocity_y(x, y, Lx, Ly):
    return 0.0

def pressure(x, y, Lx, Ly):
    return p0
