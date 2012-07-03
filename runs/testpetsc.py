
import numpy as np

u0 = 0.1
x1 = 0.5
x2 = 1.5
B0 = 1.0
p0 = 0.1

#u0 = np.sqrt(5.)
#A0 = 1.E-3
#R0 = 0.3
#p0 = 1.0
#th = np.arctan(0.5)


# Current Sheet in x

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


# Current Sheet in y

#def magnetic_x(x, y, Lx, Ly):
#    if x1 <= y and y <= x2:
#        return - B0
#    else:
#        return + B0
#
#def magnetic_y(x, y, Lx, Ly):
#    return 0.0
#
#def velocity_x(x, y, Lx, Ly):
#    return 0.0
#
#def velocity_y(x, y, Lx, Ly):
#    return u0 * np.sin(np.pi * x)


# Stationary

#def magnetic_x(x, y, Lx, Ly):
#    return + B0
#
#def magnetic_y(x, y, Lx, Ly):
#    return - B0
#
#def velocity_x(x, y, Lx, Ly):
#    return - u0
#
#def velocity_y(x, y, Lx, Ly):
#    return + u0


# Advection Loop

#def magnetic_x(x, y, Lx, Ly):
#    return - A0 * y / np.sqrt(x**2 + y**2) 
#
#def magnetic_y(x, y, Lx, Ly):
#    return + A0 * x / np.sqrt(x**2 + y**2) 
#
#def velocity_x(x, y, Lx, Ly):
#    return u0 * np.cos(th)
#
#def velocity_y(x, y, Lx, Ly):
#    return u0 * np.sin(th)


def pressure(x, y, Lx, Ly):
    return p0
