
import numpy as np

va = 1.0
u0 = 0.1
x1 = 0.5
x2 = 1.5
B0 = 1.0
p0 = 0.1
c  = 5.

Bx0 = B0
By0 = B0 * 0.5

k = 2.

#u0 = np.sqrt(5.)
#A0 = 1.E-3
#R0 = 0.3
#p0 = 1.0
#th = np.arctan(0.5)


## Wave
#
#def magnetic_x(x, y, hx, hy):
#    return Bx0
#
#def magnetic_y(x, y, hx, hy):
#    return By0 * np.cos(np.pi*x)
#
#def velocity_x(x, y, hx, hy):
#    return 0.0
#
#def velocity_y(x, y, hx, hy):
#    return va * np.cos(np.pi * x)


## Current Sheet in x
#
#def magnetic_x(x, y, hx, hy):
#    return 0.0
#
#def magnetic_y(x, y, hx, hy):
#    if x1 <= x and x <= x2:
#        return - B0
#    else:
#        return + B0
##    return np.tanh(c*(x-x1)) - np.tanh(c*(x-x2))
#
#def velocity_x(x, y, hx, hy):
#    return u0 * np.sin(np.pi * y)
#
#def velocity_y(x, y, hx, hy):
#    return 0.0


# Current Sheet in y

#def magnetic_x(x, y, hx, hy):
#    if x1 <= y and y <= x2:
#        return - B0
#    else:
#        return + B0
#
#def magnetic_y(x, y, hx, hy):
#    return 0.0
#
#def velocity_x(x, y, hx, hy):
#    return 0.0
#
#def velocity_y(x, y, hx, hy):
#    return u0 * np.sin(np.pi * x)


# Stationary

#def magnetic_x(x, y, hx, hy):
#    return + B0
#
#def magnetic_y(x, y, hx, hy):
#    return - B0
#
#def velocity_x(x, y, hx, hy):
#    return - u0
#
#def velocity_y(x, y, hx, hy):
#    return + u0


# Advection Loop

#def magnetic_x(x, y, hx, hy):
#    return - A0 * y / np.sqrt(x**2 + y**2) 
#
#def magnetic_y(x, y, hx, hy):
#    return + A0 * x / np.sqrt(x**2 + y**2) 
#
#def velocity_x(x, y, hx, hy):
#    return u0 * np.cos(th)
#
#def velocity_y(x, y, hx, hy):
#    return u0 * np.sin(th)


def pressure(x, y, hx, hy):
    return p0
