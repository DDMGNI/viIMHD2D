'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

from config import Config

from PETSc_MHD_NL         import PETScSolver
#from PETSc_MHD_NL_DF         import PETScSolver
from PETSc_MHD_NL_Function import PETScFunction
#from PETSc_MHD_NL_DF_Function import PETScFunction



class petscMHD2D(object):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # load run config file
        cfg = Config(cfgfile)
        
        # timestep setup
        self.ht    = cfg['grid']['ht']              # timestep size
        self.nt    = cfg['grid']['nt']              # number of timesteps
        self.nsave = cfg['io']['nsave']             # save only every nsave'th timestep
        self.omega = 0.1                            # relaxation parameter
        
        # grid setup
        nx   = cfg['grid']['nx']                    # number of points in x
        ny   = cfg['grid']['ny']                    # number of points in y
        Lx   = cfg['grid']['Lx']                    # spatial domain in x
        x1   = cfg['grid']['x1']                    # 
        x2   = cfg['grid']['x2']                    # 
        
        Ly   = cfg['grid']['Ly']                    # spatial domain in y
        y1   = cfg['grid']['y1']                    # 
        y2   = cfg['grid']['y2']                    # 
        
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
        
        
        self.time = PETSc.Vec().createMPI(1, PETSc.DECIDE, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol', cfg['solver']['petsc_residual'])
#        OptDB.setValue('ksp_max_it', 100)
        OptDB.setValue('ksp_max_it', 200)
#        OptDB.setValue('ksp_max_it', 1000)
#        OptDB.setValue('ksp_max_it', 2000)

#        OptDB.setValue('ksp_monitor', '')
#        OptDB.setValue('log_info', '')
#        OptDB.setValue('log_summary', '')
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[nx, ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        
        # create DA (dof = 4 for Bx, By, Vx, Vy)
        self.da4 = PETSc.DA().create(dim=2, dof=4,
                                     sizes=[nx, ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=1,
                                     stencil_type='box')
        
        
        # create DA (dof = 5 for Bx, By, Vx, Vy, P)
        self.da5 = PETSc.DA().create(dim=2, dof=5,
                                     sizes=[nx, ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=1,
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
        
        self.da4.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.da5.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.dax.setUniformCoordinates(xmin=x1, xmax=x2)
        
        self.day.setUniformCoordinates(xmin=y1, xmax=y2)
        
        
        # create solution and RHS vector
        self.x  = self.da4.createGlobalVec()
        self.b  = self.da4.createGlobalVec()
        self.u  = self.da5.createGlobalVec()
        
        # create residual vectors
        self.ru = self.da5.createGlobalVec()
        self.rx = self.da4.createGlobalVec()
        self.rp = self.da1.createGlobalVec()
        
        # create global RK4 vectors
        self.X1 = self.da4.createGlobalVec()
        self.X2 = self.da4.createGlobalVec()
        self.X3 = self.da4.createGlobalVec()
        self.X4 = self.da4.createGlobalVec()
        
        # create local RK4 vectors
        self.localX  = self.da4.createLocalVec()
        self.localX1 = self.da4.createLocalVec()
        self.localX2 = self.da4.createLocalVec()
        self.localX3 = self.da4.createLocalVec()
        self.localX4 = self.da4.createLocalVec()
        
        # create vectors for magnetic and velocity field
        self.Bx = self.da1.createGlobalVec()
        self.By = self.da1.createGlobalVec()
        self.Vx = self.da1.createGlobalVec()
        self.Vy = self.da1.createGlobalVec()
        self.P  = self.da1.createGlobalVec()
        
        # set variable names
        self.x.setName('solver_x')
        self.b.setName('solver_b')
        
        self.Bx.setName('Bx')
        self.By.setName('By')
        self.Vx.setName('Vx')
        self.Vy.setName('Vy')
        self.P.setName('P')
        
        
        # create Matrix object
        self.petsc_matrix   = PETScSolver  (self.da1, self.da4, nx, ny, self.ht, self.hx, self.hy, self.omega)
        self.petsc_function = PETScFunction(self.da1, self.da5, nx, ny, self.ht, self.hx, self.hy)
        
        # create sparse matrix
        self.mat = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], comm=PETSc.COMM_WORLD)
        self.mat.setPythonContext(self.petsc_matrix)
        self.mat.setUp()
        
        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.mat)
        self.ksp.setType(cfg['solver']['petsc_ksp_type'])
        self.ksp.setInitialGuessNonzero(True)
        
        self.pc = self.ksp.getPC()
        self.pc.setType(cfg['solver']['petsc_pc_type'])
        
        
#        # create Preconditioner matrix and solver
#        self.pc_mat = PETScPreconditioner(self.da1, self.da4, self.P, nx, ny, self.ht, self.hx, self.hy)
#        
#        # create sparse matrix
#        self.pc_A = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], comm=PETSc.COMM_WORLD)
#        self.pc_A.setPythonContext(self.pc_mat)
#        self.pc_A.setUp()
#        
#        # create linear solver and preconditioner
#        self.pc_ksp = PETSc.KSP().create()
#        self.pc_ksp.setFromOptions()
#        self.pc_ksp.setOperators(self.pc_A)
#        self.pc_ksp.setType(cfg['solver']['petsc_ksp_type'])
#        self.pc_ksp.setInitialGuessNonzero(True)
#        
#        self.pc_pc = self.pc_ksp.getPC()
#        self.pc_pc.setType('none')
#        
#        
#        # create Arakawa solver object
#        self.mhd_rk4 = PETScRK4(self.da4, nx, ny, self.ht, self.hx, self.hy)
        
        
        # set initial data
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        coords = self.da1.getCoordinateDA().getVecArray(self.da1.getCoordinates())
        
#        print
#        print(self.hx)
#        print(coords[1,0][0] - coords[0,0][0])
#        print
#        print(self.hy)
#        print(coords[0,1][1] - coords[0,0][1])
#        print
#        print(Lx)
#        print(coords[-1,0][0]+self.hx)
#        print
#        print(Ly)
#        print(coords[0,-1][1]+self.hy)
#        print
        
        x_arr  = self.da4.getVecArray(self.x)
        Bx_arr = self.da1.getVecArray(self.Bx)
        By_arr = self.da1.getVecArray(self.By)
        Vx_arr = self.da1.getVecArray(self.Vx)
        Vy_arr = self.da1.getVecArray(self.Vy)
        P_arr  = self.da1.getVecArray(self.P)
        
        
        if cfg['initial_data']['magnetic_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['magnetic_python'], globals(), locals(), ['magnetic_x', 'magnetic_y'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Bx_arr[i,j] = init_data.magnetic_x(coords[i,j][0], coords[i,j][1], Lx, Ly) 
                    By_arr[i,j] = init_data.magnetic_y(coords[i,j][0], coords[i,j][1], Lx, Ly) 
        
        else:
            Bx_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']            
            By_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']            
            
            
        if cfg['initial_data']['velocity_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['velocity_python'], globals(), locals(), ['velocity_x', 'velocity_y'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Vx_arr[i,j] = init_data.velocity_x(coords[i,j][0], coords[i,j][1], Lx, Ly) 
                    Vy_arr[i,j] = init_data.velocity_y(coords[i,j][0], coords[i,j][1], Lx, Ly) 
        
        else:
            Vx_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']            
            Vy_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']            
            
        
        if cfg['initial_data']['pressure_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['pressure_python'], globals(), locals(), ['pressure', ''], 0)
            
        for i in range(xs, xe):
            for j in range(ys, ye):
                P_arr[i,j] = init_data.pressure(coords[i,j][0], coords[i,j][1], Lx, Ly)
        
        
        # copy distribution function to solution vector
        x_arr[xs:xe, ys:ye, 0] = Bx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 1] = By_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 2] = Vx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 3] = Vy_arr[xs:xe, ys:ye]
        
        # copy 4D solution vector to 5D solution vector
        self.copy_solution_to_u()
        
        # update solution history
        self.petsc_matrix.update_history(self.x, self.P)
        self.petsc_function.update_history(self.u)
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.Viewer().createHDF5(cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.HDF5PushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        coords_x.setName('x')
        coords_y.setName('y')
        
        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        
        # write initial data to hdf5 file
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.hdf5_viewer(self.time)
        
#        self.hdf5_viewer(self.x)
#        self.hdf5_viewer(self.b)
        
        self.hdf5_viewer(self.Bx)
        self.hdf5_viewer(self.By)
        self.hdf5_viewer(self.Vx)
        self.hdf5_viewer(self.Vy)
        self.hdf5_viewer(self.P)
        
        
    
    def __del__(self):
#        if self.hdf5_viewer != None:
#            del self.hdf5_viewer
        pass
        
    
    
    def run(self):
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
            
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            # calculate initial guess
#            self.calculate_initial_guess()
            
            
            # calculate and print initial residual
            self.petsc_function.matrix_mult(self.u, self.ru)
            residual0 = self.ru.norm()
#            residual  = residual0
            norm0 = self.u.norm()
            norm  = norm0
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print
                print("   initial residual = %22.16E " % (residual0) )
                print
            
            
            # number of iterations
            nlin    = 0
            nnonlin = 0
            
            # start iteration
            while True:
                # update previous iteration
                self.petsc_matrix.update_previous(self.x, self.P)
                
                # build RHS and solve
                self.petsc_matrix.formRHS(self.b)
                self.ksp.solve(self.b, self.x)
                self.petsc_matrix.pressure(self.x, self.P)

                # calculate residual
                self.copy_solution_to_u()
                self.petsc_function.matrix_mult(self.u, self.ru)
                
                ru_arr  = self.da5.getVecArray(self.ru)
                rx_arr  = self.da4.getVecArray(self.rx)
                rp_arr  = self.da1.getVecArray(self.rp)
                
                rx_arr[xs:xe, ys:ye, :] = ru_arr[xs:xe, ys:ye, 0:4]
                rp_arr[xs:xe, ys:ye]    = ru_arr[xs:xe, ys:ye, 4  ]
                
                residual_u = self.ru.norm()
                residual_x = self.rx.norm()
                residual_p = self.rp.norm()
                
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("   it_lin = %5i,   n_iter = %5i,   res_lin = %16.10E,   res_nlin = %16.10E " % (nnonlin+1, self.ksp.getIterationNumber(), self.ksp.getResidualNorm(), residual_u) )
                    print("                                       res_x   = %16.10E,   res_p    = %16.10E " % (residual_x, residual_p) )
                
                
                # count iterations
                nnonlin += 1
                nlin    += self.ksp.getIterationNumber()
            
#                self.copy_solution_to_u()
                normp = norm
                norm  = self.u.norm()
                
#                if abs(residualp - residual) / residual0 < 1.E-10:
#                 # or abs(normp - norm) / norm0 < 1.E-5:
                if residual_u < 1.E-3:
                    break
                
            
            # copy solution to 5D vector
#            self.copy_solution_to_u()
        
            # update history
            self.petsc_matrix.update_history(self.x, self.P)
            self.petsc_function.update_history(self.u)
            
            # save to hdf5 file
            if itime % self.nsave == 0 or itime == self.nt + 1:
                self.save_to_hdf5()
            
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("   Solver:  %5i nonlinear iterations " % (nnonlin) )
                print("            %5i linear    iterations " % (nlin)    )
                print("   Function Norm = %24.16E" % (residual_u) )
           
    
    def copy_solution_to_u(self):
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        u_arr  = self.da5.getVecArray(self.u)
        x_arr  = self.da4.getVecArray(self.x)
        P_arr  = self.da1.getVecArray(self.P)
        
        u_arr[xs:xe, ys:ye, 0:4] = x_arr[xs:xe, ys:ye, :]
        u_arr[xs:xe, ys:ye, 4  ] = P_arr[xs:xe, ys:ye]
    
    
    
    
    def save_to_hdf5(self):
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        # copy solution to B and V vectors
        x_arr  = self.da4.getVecArray(self.x)
        Bx_arr = self.da1.getVecArray(self.Bx)
        By_arr = self.da1.getVecArray(self.By)
        Vx_arr = self.da1.getVecArray(self.Vx)
        Vy_arr = self.da1.getVecArray(self.Vy)
        
        Bx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 0]
        By_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 1]
        Vx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 2]
        Vy_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 3]
        
        
        # save timestep
        self.hdf5_viewer.HDF5SetTimestep(self.hdf5_viewer.HDF5GetTimestep() + 1)
        self.hdf5_viewer(self.time)
        
        # save RHS and solution vector
#        self.hdf5_viewer(self.x)
#        self.hdf5_viewer(self.b)
        
        # save data
        self.hdf5_viewer(self.Bx)
        self.hdf5_viewer(self.By)
        self.hdf5_viewer(self.Vx)
        self.hdf5_viewer(self.Vy)
        self.hdf5_viewer(self.P)
    
    
    def calculate_initial_guess(self):
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        # explicit predictor for Bx, By, Vx, Vy
        self.rk4(self.x)
        
#        # calculate initial guess for total pressure
#        self.petsc_matrix.formRHSPoisson(self.Pb, self.x)
#        self.poisson_ksp.solve(self.Pb, self.P)
#        
#        P_arr = self.da1.getVecArray(self.P)
#        x_arr = self.da4.getVecArray(self.x)
#        
#        x_arr[xs:xe, ys:ye, 4] = P_arr[xs:xe, ys:ye]
#        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("   Poisson: %5i iterations,   residual = %24.16E " % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
        
        
        # calculate initial guess for total pressure
        self.petsc_matrix.formRHSPoisson(self.Pb, self.x)
        self.poisson_ksp.solve(self.Pb, self.P)
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Poisson: %5i iterations,   residual = %24.16E " % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
        
        # precondition V and B
        self.pc_mat.formRHS(self.b)
        self.pc_ksp.solve(self.b, self.x)

        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Precon : %5i iterations,   residual = %24.16E " % (self.pc_ksp.getIterationNumber(), self.pc_ksp.getResidualNorm()) )
        
        # precondition P
#        self.petsc_matrix.formRHSPoisson(self.Pb, self.x)
#        self.poisson_ksp.solve(self.Pb, self.P)
#        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("   Poisson: %5i iterations,   residual = %24.16E " % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
        
        # copy pressure to solution vector
        P_arr = self.da1.getVecArray(self.P)
        x_arr = self.da4.getVecArray(self.x)
        
        x_arr[xs:xe, ys:ye, 4] = P_arr[xs:xe, ys:ye]
        
        
            
    def rk4(self, X):
        
        self.da4.globalToLocal(X, self.localX)
        x  = self.da4.getVecArray(self.localX)[...]
        x1 = self.da4.getVecArray(self.X1)[...]
        self.petsc_matrix.timestep(x, x1)
        
        self.da4.globalToLocal(self.X1, self.localX1)
        x1 = self.da4.getVecArray(self.localX1)[...]
        x2 = self.da4.getVecArray(self.X2)[...]
        self.petsc_matrix.timestep(x + 0.5 * self.ht * x1, x2)
        
        self.da4.globalToLocal(self.X2, self.localX2)
        x2 = self.da4.getVecArray(self.localX2)[...]
        x3 = self.da4.getVecArray(self.X3)[...]
        self.petsc_matrix.timestep(x + 0.5 * self.ht * x2, x3)
        
        self.da4.globalToLocal(self.X3, self.localX3)
        x3 = self.da4.getVecArray(self.localX3)[...]
        x4 = self.da4.getVecArray(self.X4)[...]
        self.petsc_matrix.timestep(x + 1.0 * self.ht * x3, x4)
        
        x  = self.da4.getVecArray(X)[...]
        x1 = self.da4.getVecArray(self.X1)[...]
        x2 = self.da4.getVecArray(self.X2)[...]
        x3 = self.da4.getVecArray(self.X3)[...]
        x4 = self.da4.getVecArray(self.X4)[...]
        
        x[:,:,:] = x + self.ht * (x1 + 2.*x2 + 2.*x3 + x4) / 6.

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
    
