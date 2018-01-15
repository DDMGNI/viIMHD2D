# viIMHD2D

*Python/Cython package providing reference implementations of variational integrators for ideal and inertial magnetohydrodynamics in 2D.*

[![Project Status: Inactive](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org/#inactive)
[![License: MIT](https://img.shields.io/badge/license-MIT%20License-blue.svg)](LICENSE.md)


The code must first be built by calling `make` in the main directory.
Then the ideal MHD code can be run, e.g., in serial via

```
> python ideal_mhd2d_nonlinear_newton_snes_gmres.py examples/alfven_wave_travelling.cfg
```

and in parallel via

```
> mpiexec -n 4 python ideal_mhd2d_nonlinear_newton_snes_gmres.py examples/alfven_wave_travelling.cfg
```
The run script for the inertial MHD code is `inertial_mhd2d_nonlinear_newton_snes_gmres.py`.


## References

_Michael Kraus, Omar Maj_. Variational Integrators for Nonvariational Partial Differential Equations. Physica D: Nonlinear Phenomena, Volume 310, Pages 37-71, 2015.
[Journal](https://dx.doi.org/10.1016/j.physd.2015.08.002),
[arXiv:1412.2011](https://arxiv.org/abs/1412.2011).  

_Michael Kraus, Omar Maj_. Variational Integrators for Ideal Magnetohydrodynamics. 
[arXiv:1707.03227](https://arxiv.org/abs/1707.03227).  

_Michael Kraus_. Variational Integrators for Inertial Magnetohydrodynamics. 
In preparation.  


## License

The viIMHD2D package is licensed under the [MIT "Expat" License](LICENSE.md).
