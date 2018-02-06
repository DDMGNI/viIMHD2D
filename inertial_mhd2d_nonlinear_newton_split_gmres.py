'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time
import numpy as np 
import h5py

from config import Config

from imhd.integrators.Inertial_MHD_Nonlinear import PETScFunction
from imhd.integrators.Inertial_MHD_Euler     import PETScSolverEuler
from imhd.integrators.Inertial_MHD_Faraday   import PETScSolverFaraday
from imhd.integrators.Inertial_MHD_Poisson   import PETScSolverPoisson
from imhd.integrators.MHD_Derivatives        import MHD_Derivatives


class petscMHD2D(object):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
#        stencil = 1
        stencil = 2
        
        # load run config file
        cfg = Config(cfgfile)
        self.cfg = cfg
        cfg.set_petsc_options()

        OptDB = PETSc.Options()
        OptDB.setValue('snes_force_iteration', True)
        OptDB.setValue('pc_fieldsplit_type', 'additive')
        OptDB.setValue('pc_fieldsplit_detect_saddle_point', True)
        OptDB.setValue('fieldsplit_0_ksp_type', 'preonly')
        OptDB.setValue('fieldsplit_0_pc_type', 'lu')
        OptDB.setValue('fieldsplit_1_ksp_type', 'preonly')
        OptDB.setValue('fieldsplit_1_pc_type', 'jacobi')
        
        # timestep setup
        self.ht    = cfg['grid']['ht']              # timestep size
        self.nt    = cfg['grid']['nt']              # number of timesteps
        self.nsave = cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        nx   = cfg['grid']['nx']                    # number of points in x
        ny   = cfg['grid']['ny']                    # number of points in y
        Lx   = cfg['grid']['Lx']                    # spatial domain in x
        x1   = cfg['grid']['x1']                    # 
        x2   = cfg['grid']['x2']                    # 
        
        Ly   = cfg['grid']['Ly']                    # spatial domain in y
        y1   = cfg['grid']['y1']                    # 
        y2   = cfg['grid']['y2']                    # 
        
        self.nx = nx
        self.ny = ny
        
        if x1 != x2:
            Lx = x2-x1
        else:
            x1 = 0.0
            x2 = Lx
        
        if y1 != y2:
            Ly = y2-y1
        else:
            y1 = 0.0
            y2 = Ly
        
        
        self.hx = Lx / nx                       # gridstep size in x
        self.hy = Ly / ny                       # gridstep size in y
        
        
        # friction, viscosity and resistivity
        mu  = cfg['initial_data']['mu']                    # friction
        nu  = cfg['initial_data']['nu']                    # viscosity
        eta = cfg['initial_data']['eta']                   # resistivity
        de  = cfg['initial_data']['de']                    # electron skin depth
        
        # solver parameters
        self.atol = cfg['solver']['petsc_snes_atol']
        self.rtol = cfg['solver']['petsc_snes_rtol']
        self.stol = cfg['solver']['petsc_snes_stol']
        self.max_it = cfg['solver']['petsc_snes_max_iter']
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print()
            print("nt = %i" % (self.nt))
            print("nx = %i" % (self.nx))
            print("ny = %i" % (self.ny))
            print()
            print("ht = %e" % (self.ht))
            print("hx = %e" % (self.hx))
            print("hy = %e" % (self.hy))
            print()
            print("Lx   = %e" % (Lx))
            print("Ly   = %e" % (Ly))
            print()
            print("mu   = %e" % (mu))
            print("nu   = %e" % (nu))
            print("eta  = %e" % (eta))
            print()
        
        
        self.time = PETSc.Vec().createMPI(1, PETSc.DECIDE, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # create derivatives object
        self.derivatives = MHD_Derivatives(nx, ny, self.ht, self.hx, self.hy)
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[nx, ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=stencil,
                                    stencil_type='box')
        
        
        # create DA (dof = 2 for Bx, By or Bix, Biy)
        self.da2 = PETSc.DA().create(dim=2, dof=2,
                                    sizes=[nx, ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=stencil,
                                    stencil_type='box')
        
        
        # create DA (dof = 3 for Vx, Vy, P)
        self.da3 = PETSc.DA().create(dim=2, dof=3,
                                    sizes=[nx, ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=stencil,
                                    stencil_type='box')
        
        
        # create DA (dof = 7 for Vx, Vy, Bx, By, Bix, Biy, P)
        self.da7 = PETSc.DA().create(dim=2, dof=7,
                                     sizes=[nx, ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=stencil,
                                     stencil_type='box')
        
        
        # create DA for x grid
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[nx],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[ny],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.da2.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.da3.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.da7.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.dax.setUniformCoordinates(xmin=x1, xmax=x2)
        
        self.day.setUniformCoordinates(xmin=y1, xmax=y2)
        
        
        # create solution and RHS vectors
        self.f  = self.da7.createGlobalVec()
        self.x  = self.da7.createGlobalVec()
        self.b  = self.da7.createGlobalVec()
        
        # create vectors for magnetic and velocity field
        self.V  = self.da3.createGlobalVec()
        self.B  = self.da2.createGlobalVec()
        self.Bi = self.da2.createGlobalVec()
        
        # create RHS vectors for magnetic and velocity field
        self.FE = self.da3.createGlobalVec()
        self.FF = self.da2.createGlobalVec()
        self.FP = self.da2.createGlobalVec()
        
        # create vectors for magnetic and velocity field
        self.Bix = self.da1.createGlobalVec()
        self.Biy = self.da1.createGlobalVec()
        self.Bx  = self.da1.createGlobalVec()
        self.By  = self.da1.createGlobalVec()
        self.Vx  = self.da1.createGlobalVec()
        self.Vy  = self.da1.createGlobalVec()
        self.P   = self.da1.createGlobalVec()

        self.xcoords = self.da1.createGlobalVec()
        self.ycoords = self.da1.createGlobalVec()
        
        # create local vectors for initialisation of pressure
        self.localBix = self.da1.createLocalVec()
        self.localBiy = self.da1.createLocalVec()
        self.localBx  = self.da1.createLocalVec()
        self.localBy  = self.da1.createLocalVec()
        self.localVx  = self.da1.createLocalVec()
        self.localVy  = self.da1.createLocalVec()
        
        # set variable names
        self.Bix.setName('Bix')
        self.Biy.setName('Biy')
        self.Bx.setName('Bx')
        self.By.setName('By')
        self.Vx.setName('Vx')
        self.Vy.setName('Vy')
        self.P.setName('P')
        
        
        # create Matrix object
        self.petsc_function = PETScFunction     (self.da1, self.da7, nx, ny, self.ht, self.hx, self.hy, mu, nu, eta, de)
        self.petsc_euler    = PETScSolverEuler  (self.da3, self.da7, nx, ny, self.ht, self.hx, self.hy, mu, nu, eta, de)
        self.petsc_faraday  = PETScSolverFaraday(self.da2, self.da7, nx, ny, self.ht, self.hx, self.hy, mu, nu, eta, de)
        self.petsc_poisson  = PETScSolverPoisson(self.da2, self.da7, nx, ny, self.ht, self.hx, self.hy, mu, nu, eta, de)
        
        # create jacobians (JE: Euler, JF: Faraday, JP: Poisson)
        self.JE = self.da3.createMat()
        self.JE.setOption(self.JE.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.JE.setUp()
        
        self.JF = self.da2.createMat()
        self.JF.setOption(self.JF.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.JF.setUp()
        
        self.JP = self.da2.createMat()
        self.JP.setOption(self.JP.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.JP.setUp()
        
        
        # create nonlinear solver for Euler equation
        self.snes_euler = PETSc.SNES().create()
        self.snes_euler.setType('ksponly')
        self.snes_euler.setFunction(self.petsc_euler.snes_mult, self.FE)
        self.snes_euler.setJacobian(self.updateJacobianEuler, self.JE)
        self.snes_euler.setFromOptions()
        self.snes_euler.getKSP().setType('gmres')
        self.snes_euler.getKSP().getPC().setType('fieldsplit')
        
        # create nonlinear solver for Faraday equation
        self.snes_faraday = PETSc.SNES().create()
        self.snes_faraday.setType('ksponly')
        self.snes_faraday.setFunction(self.petsc_faraday.snes_mult, self.FF)
        self.snes_faraday.setJacobian(self.updateJacobianFaraday, self.JF)
        self.snes_faraday.setFromOptions()
        self.snes_faraday.getKSP().setType('gmres')
        self.snes_faraday.getKSP().getPC().setType('asm')
        
        OptDB.setValue('snes_lag_preconditioner', -1)

        # create linear solver for Poisson equation
        self.snes_poisson = PETSc.SNES().create()
        self.snes_poisson.setType('ksponly')
        self.snes_poisson.setFunction(self.petsc_poisson.snes_mult, self.FP)
        self.snes_poisson.setJacobian(self.updateJacobianPoisson, self.JP)
        self.snes_poisson.setFromOptions()
        self.snes_poisson.getKSP().setType('cg')
#        self.snes_poisson.getKSP().getPC().setType('hypre')
        self.snes_poisson.getKSP().getPC().setType('lu')
        
        
        # set initial data
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        xc_arr = self.da1.getVecArray(self.xcoords)
        yc_arr = self.da1.getVecArray(self.ycoords)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                xc_arr[i,j] = x1 + i*self.hx
                yc_arr[i,j] = y1 + j*self.hy
        
        
        if cfg['io']['hdf5_input'] != None:
            hdf5_filename = self.cfg["io"]["hdf5_input"]
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Input:  %s" % hdf5_filename)
                
            hdf5in = h5py.File(hdf5_filename, "r", driver="mpio", comm=PETSc.COMM_WORLD.tompi4py())
            
#             assert self.nx == hdf5in.attrs["grid.nx"]
#             assert self.ny == hdf5in.attrs["grid.ny"]
#             assert self.hx == hdf5in.attrs["grid.hx"]
#             assert self.hy == hdf5in.attrs["grid.hy"]
#             assert self.Lx == hdf5in.attrs["grid.Lx"]
#             assert self.Ly == hdf5in.attrs["grid.Ly"]
#             
#             assert self.de == hdf5in.attrs["initial_data.skin_depth"]
            
            timestep = len(hdf5in["t"][...].flatten()) - 1
            
            hdf5in.close()
            
            hdf5_viewer = PETSc.ViewerHDF5().create(cfg['io']['hdf5_input'],
                                              mode=PETSc.Viewer.Mode.READ,
                                              comm=PETSc.COMM_WORLD)
            
            hdf5_viewer.setTimestep(timestep)

            self.Bix.load(hdf5_viewer)
            self.Biy.load(hdf5_viewer)
            self.Bx.load(hdf5_viewer)
            self.By.load(hdf5_viewer)
            self.Vx.load(hdf5_viewer)
            self.Vy.load(hdf5_viewer)
            self.P.load(hdf5_viewer)
            
            hdf5_viewer.destroy()


            # copy modified magnetic induction to solution vector
            Bx_arr  = self.da1.getVecArray(self.Bx)
            By_arr  = self.da1.getVecArray(self.By)
            Vx_arr  = self.da1.getVecArray(self.Vx)
            Vy_arr  = self.da1.getVecArray(self.Vy)
            Bix_arr = self.da1.getVecArray(self.Bix)
            Biy_arr = self.da1.getVecArray(self.Biy)
            P_arr   = self.da1.getVecArray(self.P)
        
            x_arr = self.da7.getVecArray(self.x)
            x_arr[xs:xe, ys:ye, 0] = Vx_arr [xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 1] = Vy_arr [xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 2] = Bx_arr [xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 3] = By_arr [xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 4] = Bix_arr[xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 5] = Biy_arr[xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 6] = P_arr  [xs:xe, ys:ye]

            v_arr = self.da3.getVecArray(self.V)
            v_arr[xs:xe, ys:ye, 0] = Vx_arr[xs:xe, ys:ye]
            v_arr[xs:xe, ys:ye, 1] = Vy_arr[xs:xe, ys:ye]
            v_arr[xs:xe, ys:ye, 2] = P_arr [xs:xe, ys:ye]

            b_arr = self.da2.getVecArray(self.B)
            b_arr[xs:xe, ys:ye, 0] = Bx_arr[xs:xe, ys:ye]
            b_arr[xs:xe, ys:ye, 1] = By_arr[xs:xe, ys:ye]

            b_arr = self.da2.getVecArray(self.Bi)
            b_arr[xs:xe, ys:ye, 0] = Bix_arr[xs:xe, ys:ye]
            b_arr[xs:xe, ys:ye, 1] = Biy_arr[xs:xe, ys:ye]
            
        else:
        
            Bx_arr = self.da1.getVecArray(self.Bx)
            By_arr = self.da1.getVecArray(self.By)
            Vx_arr = self.da1.getVecArray(self.Vx)
            Vy_arr = self.da1.getVecArray(self.Vy)
        
            xc_arr = self.da1.getVecArray(self.xcoords)
            yc_arr = self.da1.getVecArray(self.ycoords)
            
            if cfg['initial_data']['magnetic_python'] != None:
                init_data = __import__("examples." + cfg['initial_data']['magnetic_python'], globals(), locals(), ['magnetic_x', 'magnetic_y'], 0)
                
                for i in range(xs, xe):
                    for j in range(ys, ye):
                        Bx_arr[i,j] = init_data.magnetic_x(xc_arr[i,j], yc_arr[i,j] + 0.5 * self.hy, self.hx, self.hy)
                        By_arr[i,j] = init_data.magnetic_y(xc_arr[i,j] + 0.5 * self.hx, yc_arr[i,j], self.hx, self.hy)
            
            else:
                Bx_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']
                By_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']
                
                
            if cfg['initial_data']['velocity_python'] != None:
                init_data = __import__("examples." + cfg['initial_data']['velocity_python'], globals(), locals(), ['velocity_x', 'velocity_y'], 0)
                
                for i in range(xs, xe):
                    for j in range(ys, ye):
                        Vx_arr[i,j] = init_data.velocity_x(xc_arr[i,j], yc_arr[i,j] + 0.5 * self.hy, self.hx, self.hy)
                        Vy_arr[i,j] = init_data.velocity_y(xc_arr[i,j] + 0.5 * self.hx, yc_arr[i,j], self.hx, self.hy)
            
            else:
                Vx_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']
                Vy_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']
                
            
            if cfg['initial_data']['pressure_python'] != None:
                init_data = __import__("examples." + cfg['initial_data']['pressure_python'], globals(), locals(), ['pressure', ''], 0)
        
        
            # Fourier Filtering
            from scipy.fftpack import rfft, irfft
            
            nfourier_x = cfg['initial_data']['nfourier_Bx']
            nfourier_y = cfg['initial_data']['nfourier_By']
              
            if nfourier_x >= 0 or nfourier_y >= 0:
                print("Fourier Filtering B")
                
                # obtain whole Bx vector everywhere
                scatter, Xglobal = PETSc.Scatter.toAll(self.Bx)
                
                scatter.begin(self.Bx, Xglobal, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                scatter.end  (self.Bx, Xglobal, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                
                petsc_indices = self.da1.getAO().app2petsc(np.arange(self.nx*self.ny, dtype=np.int32))
                
                BxTmp = Xglobal.getValues(petsc_indices).copy().reshape((self.ny, self.nx)).T
                
                scatter.destroy()
                Xglobal.destroy()
                
                # obtain whole By vector everywhere
                scatter, Xglobal = PETSc.Scatter.toAll(self.By)
                
                scatter.begin(self.By, Xglobal, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                scatter.end  (self.By, Xglobal, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                
                petsc_indices = self.da1.getAO().app2petsc(np.arange(self.nx*self.ny, dtype=np.int32))
                
                ByTmp = Xglobal.getValues(petsc_indices).copy().reshape((self.ny, self.nx)).T
                
                scatter.destroy()
                Xglobal.destroy()
                
                
                if nfourier_x >= 0:
                    # compute FFT, cut, compute inverse FFT
                    BxFft = rfft(BxTmp, axis=1)
                    ByFft = rfft(ByTmp, axis=1)
                
                    BxFft[:,nfourier_x+1:] = 0.
                    ByFft[:,nfourier_x+1:] = 0.
                    
                    BxTmp = irfft(BxFft, axis=1)
                    ByTmp = irfft(ByFft, axis=1)


                if nfourier_y >= 0:
                    BxFft = rfft(BxTmp, axis=0)
                    ByFft = rfft(ByTmp, axis=0)
                
                    BxFft[nfourier_y+1:,:] = 0.
                    ByFft[nfourier_y+1:,:] = 0.

                    BxTmp = irfft(BxFft, axis=0)
                    ByTmp = irfft(ByFft, axis=0)
                
                
                Bx_arr = self.da1.getVecArray(self.Bx)
                By_arr = self.da1.getVecArray(self.By)
                
                Bx_arr[:,:] = BxTmp[xs:xe, ys:ye]
                By_arr[:,:] = ByTmp[xs:xe, ys:ye]
                
            
            Bx_arr = self.da1.getVecArray(self.Bx)
            By_arr = self.da1.getVecArray(self.By)
            Vx_arr = self.da1.getVecArray(self.Vx)
            Vy_arr = self.da1.getVecArray(self.Vy)
            
            x_arr = self.da7.getVecArray(self.x)
            v_arr = self.da3.getVecArray(self.V)
            b_arr = self.da2.getVecArray(self.B)
            x_arr[xs:xe, ys:ye, 0] = Vx_arr[xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 1] = Vy_arr[xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 2] = Bx_arr[xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 3] = By_arr[xs:xe, ys:ye]
            v_arr[xs:xe, ys:ye, 0] = Vx_arr[xs:xe, ys:ye]
            v_arr[xs:xe, ys:ye, 1] = Vy_arr[xs:xe, ys:ye]
            b_arr[xs:xe, ys:ye, 0] = Bx_arr[xs:xe, ys:ye]
            b_arr[xs:xe, ys:ye, 1] = By_arr[xs:xe, ys:ye]
            
            # compure generalised magnetic induction
            self.da1.globalToLocal(self.Bx, self.localBx)
            self.da1.globalToLocal(self.By, self.localBy)
            self.da1.globalToLocal(self.Vx, self.localVx)
            self.da1.globalToLocal(self.Vy, self.localVy)
            
            Bx_arr  = self.da1.getVecArray(self.localBx)
            By_arr  = self.da1.getVecArray(self.localBy)
            Vx_arr  = self.da1.getVecArray(self.localVx)
            Vy_arr  = self.da1.getVecArray(self.localVy)
            
            Bix_arr = self.da1.getVecArray(self.Bix)
            Biy_arr = self.da1.getVecArray(self.Biy)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Bix_arr[i,j] = self.derivatives.Bix(Bx_arr[...], By_arr[...], i-xs+2, j-ys+2, de)
                    Biy_arr[i,j] = self.derivatives.Biy(Bx_arr[...], By_arr[...], i-xs+2, j-ys+2, de)
            
            # copy modified magnetic induction to solution vector
            x_arr = self.da7.getVecArray(self.x)
            b_arr = self.da2.getVecArray(self.Bi)
            x_arr[xs:xe, ys:ye, 4] = Bix_arr[xs:xe, ys:ye]
            x_arr[xs:xe, ys:ye, 5] = Biy_arr[xs:xe, ys:ye]
            b_arr[xs:xe, ys:ye, 0] = Bix_arr[xs:xe, ys:ye]
            b_arr[xs:xe, ys:ye, 1] = Biy_arr[xs:xe, ys:ye]
            
            # compute pressure
            self.da1.globalToLocal(self.Bix, self.localBix)
            self.da1.globalToLocal(self.Biy, self.localBiy)
            
            Bix_arr  = self.da1.getVecArray(self.localBix)
            Biy_arr  = self.da1.getVecArray(self.localBiy)
            
            P_arr   = self.da1.getVecArray(self.P)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    P_arr[i,j] = init_data.pressure(xc_arr[i,j] + 0.5 * self.hx, yc_arr[i,j] + 0.5 * self.hy, self.hx, self.hy) \
                               - 0.5 * 0.25 * (Bix_arr[i,j] + Bix_arr[i+1,j]) * (Bx_arr[i,j] + Bx_arr[i+1,j]) \
                               - 0.5 * 0.25 * (Biy_arr[i,j] + Biy_arr[i,j+1]) * (By_arr[i,j] + By_arr[i,j+1]) \
    #                            - 0.5 * (0.25 * (Vx_arr[i,j] + Vx_arr[i+1,j])**2 + 0.25 * (Vy_arr[i,j] + Vy_arr[i,j+1])**2)
            
            
            # copy pressure to solution vector
            x_arr = self.da7.getVecArray(self.x)
            v_arr = self.da3.getVecArray(self.V)
            x_arr[xs:xe, ys:ye, 6] = P_arr  [xs:xe, ys:ye]
            v_arr[xs:xe, ys:ye, 2] = P_arr  [xs:xe, ys:ye]
        
        
        # update solution history
        self.petsc_function.update_history(self.x)
        self.petsc_euler.update_history(self.x)
        self.petsc_faraday.update_history(self.x)
        self.petsc_poisson.update_history(self.x)
        
        
        hdf5_filename = cfg['io']['hdf5_output']
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Output: %s" % hdf5_filename)
        
        hdf5out = h5py.File(hdf5_filename, "w", driver="mpio", comm=PETSc.COMM_WORLD.tompi4py())
        
        for cfg_group in self.cfg:
            for cfg_item in self.cfg[cfg_group]:
                if self.cfg[cfg_group][cfg_item] != None:
                    value = self.cfg[cfg_group][cfg_item]
                else:
                    value = ""
                    
                hdf5out.attrs[cfg_group + "." + cfg_item] = value
        
#         if self.cfg["initial_data"]["python"] != None and self.cfg["initial_data"]["python"] != "":
#             python_input = open("runs/" + self.cfg['initial_data']['python'] + ".py", 'r')
#             python_file = python_input.read()
#             python_input.close()
#         else:
#             python_file = ""
            
#         hdf5out.attrs["initial_data.python_file"] = python_file
        hdf5out.close()
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.ViewerHDF5().create(hdf5_filename,
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.pushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        coords_x.setName('x')
        coords_y.setName('y')
        
        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        
        # write initial data to hdf5 file
        self.hdf5_viewer.setTimestep(0)
        self.save_hdf5_vectors()
        
        
    
    def __del__(self):
        self.hdf5_viewer.destroy()
        self.snes_euler.destroy()
        self.snes_faraday.destroy()
        self.snes_poisson.destroy()
        
        self.JE.destroy()
        self.JF.destroy()
        self.JP.destroy()
        
        self.da1.destroy()
        self.da2.destroy()
        self.da3.destroy()
        self.da7.destroy()
        self.dax.destroy()
        self.day.destroy()

    
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_function.update_previous(X)
        self.petsc_function.formMat(J)
    
    def updateJacobianEuler(self, snes, X, J, P):
        self.petsc_euler.update_previous_V(X)
        self.petsc_euler.formMat(J)
    
    def updateJacobianFaraday(self, snes, X, J, P):
        self.petsc_faraday.update_previous_Bi(X)
        self.petsc_faraday.formMat(J)
    
    def updateJacobianPoisson(self, snes, X, J, P):
        self.petsc_poisson.update_previous_B(X)
        self.petsc_poisson.formMat(J)
    
    
    def run(self):
        
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                print()
                self.time.setValue(0, self.ht*itime)
            
            # update Jacobian
            self.update_jacobian = True
            
            # build RHS and calculate norm
            self.petsc_function.matrix_mult(self.x, self.b)
            normi = self.b.norm()
            norm1 = normi
            
            
            for n in range(self.max_it):
                # store previous norm
                norm0 = norm1
            
                # solve Euler equation
                self.petsc_euler.update_previous(self.x)
                self.snes_euler.solve(None, self.V)
                self.copy_V_to_X()
#                     
#                 # solve Faraday equation
                self.petsc_faraday.update_previous(self.x)
                self.snes_faraday.solve(None, self.Bi)
                self.copy_Bi_to_X()
                    
                # solve Poisson equation
                self.petsc_poisson.update_previous(self.x)
                self.snes_poisson.solve(None, self.B)
                self.copy_B_to_X()
                
                # build RHS and calculate norm
                self.petsc_function.matrix_mult(self.x, self.b)
                norm1 = self.b.norm()
                
                # output some solver info
                if PETSc.COMM_WORLD.getRank() == 0:
    #                 print("  Euler Solver:      %5i iterations" % (neuler) )
    #                 print("  Faraday Solver:    %5i iterations" % (nfaraday) )
    #                 print("  Poisson Solver:    %5i iterations" % (npoisson) )
                    print("  Nonlinear Solver:  %5i iterations,   funcnorm = %24.16E,   rel = %24.16E,   suc = %24.16E" % (n, norm1, abs(norm1/normi), abs(norm1-norm0)) )
                
                # check breakout criteria
                if norm1 < self.atol or abs(norm1/normi) < self.rtol or abs(norm1-norm0) < self.stol:
                    break
                    
            
            # update history
            self.petsc_function.update_history(self.x)
            self.petsc_euler.update_history(self.x)
            self.petsc_faraday.update_history(self.x)
            self.petsc_poisson.update_history(self.x)
            
            # save to hdf5 file
            if itime % self.nsave == 0 or itime == self.nt + 1:
                self.save_to_hdf5()
            

    def copy_V_to_X(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # copy solution from V to x vector
        x_arr = self.da7.getVecArray(self.x)
        v_arr = self.da3.getVecArray(self.V)
        
        x_arr[xs:xe, ys:ye, 0] = v_arr[xs:xe, ys:ye, 0]
        x_arr[xs:xe, ys:ye, 1] = v_arr[xs:xe, ys:ye, 1]
        x_arr[xs:xe, ys:ye, 6] = v_arr[xs:xe, ys:ye, 2]

    def copy_B_to_X(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # copy solution from B to x vector
        x_arr = self.da7.getVecArray(self.x)
        b_arr = self.da2.getVecArray(self.B)
        
        x_arr[xs:xe, ys:ye, 2] = b_arr[xs:xe, ys:ye, 0]
        x_arr[xs:xe, ys:ye, 3] = b_arr[xs:xe, ys:ye, 1]

    def copy_Bi_to_X(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # copy solution from Bi to x vector
        x_arr = self.da7.getVecArray(self.x)
        b_arr = self.da2.getVecArray(self.Bi)
        
        x_arr[xs:xe, ys:ye, 4] = b_arr[xs:xe, ys:ye, 0]
        x_arr[xs:xe, ys:ye, 5] = b_arr[xs:xe, ys:ye, 1]

    
    
    def save_to_hdf5(self):
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # copy solution to B and V vectors
        x_arr   = self.da7.getVecArray(self.x)
        Bix_arr = self.da1.getVecArray(self.Bix)
        Biy_arr = self.da1.getVecArray(self.Biy)
        Bx_arr  = self.da1.getVecArray(self.Bx)
        By_arr  = self.da1.getVecArray(self.By)
        Vx_arr  = self.da1.getVecArray(self.Vx)
        Vy_arr  = self.da1.getVecArray(self.Vy)
        P_arr   = self.da1.getVecArray(self.P)
        
        Vx_arr [xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 0]
        Vy_arr [xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 1]
        Bx_arr [xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 2]
        By_arr [xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 3]
        Bix_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 4]
        Biy_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 5]
        P_arr  [xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 6]
        
        
        # save timestep
        self.hdf5_viewer.setTimestep(self.hdf5_viewer.getTimestep() + 1)
        
        # save data
        self.save_hdf5_vectors()
    
    
    def save_hdf5_vectors(self):
        self.hdf5_viewer(self.time)
        self.hdf5_viewer(self.Bix)
        self.hdf5_viewer(self.Biy)
        self.hdf5_viewer(self.Bx)
        self.hdf5_viewer(self.By)
        self.hdf5_viewer(self.Vx)
        self.hdf5_viewer(self.Vy)
        self.hdf5_viewer(self.P)
        

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
    
