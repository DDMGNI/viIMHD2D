

import  numpy as np
cimport numpy as np



class pyMHD2D_RK4(object):
    
    def __init__(self, nx, ny, ht, hx, hy):
        self.nx = nx
        self.ny = ny
        self.ht = ht
        self.hx = hx
        self.hy = hy
    
    
    def rk4(self, np.ndarray[np.float64_t, ndim=2] Bx,
                  np.ndarray[np.float64_t, ndim=2] By,
                  np.ndarray[np.float64_t, ndim=2] Vx,
                  np.ndarray[np.float64_t, ndim=2] Vy):
        
        cdef np.float64_t ht = self.ht
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx1 = np.empty_like(Bx)
        cdef np.ndarray[np.float64_t, ndim=2] Bx2 = np.empty_like(Bx)
        cdef np.ndarray[np.float64_t, ndim=2] Bx3 = np.empty_like(Bx)
        cdef np.ndarray[np.float64_t, ndim=2] Bx4 = np.empty_like(Bx)
        
        cdef np.ndarray[np.float64_t, ndim=2] By1 = np.empty_like(By)
        cdef np.ndarray[np.float64_t, ndim=2] By2 = np.empty_like(By)
        cdef np.ndarray[np.float64_t, ndim=2] By3 = np.empty_like(By)
        cdef np.ndarray[np.float64_t, ndim=2] By4 = np.empty_like(By)
        
        cdef np.ndarray[np.float64_t, ndim=2] Vx1 = np.empty_like(Vx)
        cdef np.ndarray[np.float64_t, ndim=2] Vx2 = np.empty_like(Vx)
        cdef np.ndarray[np.float64_t, ndim=2] Vx3 = np.empty_like(Vx)
        cdef np.ndarray[np.float64_t, ndim=2] Vx4 = np.empty_like(Vx)
        
        cdef np.ndarray[np.float64_t, ndim=2] Vy1 = np.empty_like(Vy)
        cdef np.ndarray[np.float64_t, ndim=2] Vy2 = np.empty_like(Vy)
        cdef np.ndarray[np.float64_t, ndim=2] Vy3 = np.empty_like(Vy)
        cdef np.ndarray[np.float64_t, ndim=2] Vy4 = np.empty_like(Vy)
        
        
        self.timestep(Bx, By, Vx, Vy, Bx1, By1, Vx1, Vy1)
        
        self.timestep(Bx + 0.5 * self.ht * Bx1,
                      By + 0.5 * self.ht * By1,
                      Vx + 0.5 * self.ht * Vx1,
                      Vy + 0.5 * self.ht * Vy1,
                      Bx2, By2, Vx2, Vy2)
        
        self.timestep(Bx + 0.5 * self.ht * Bx2,
                      By + 0.5 * self.ht * By2,
                      Vx + 0.5 * self.ht * Vx2,
                      Vy + 0.5 * self.ht * Vy2,
                      Bx3, By3, Vx3, Vy3)
        
        self.timestep(Bx + 1.0 * self.ht * Bx3,
                      By + 1.0 * self.ht * By3,
                      Vx + 1.0 * self.ht * Vx3,
                      Vy + 1.0 * self.ht * Vy3,
                      Bx4, By4, Vx4, Vy4)
        
        
        Bx[:,:] += (Bx1 + 2.*Bx2 + 2.*Bx3 + Bx4) * self.ht / 6.
        By[:,:] += (By1 + 2.*By2 + 2.*By3 + By4) * self.ht / 6.
        Vx[:,:] += (Vx1 + 2.*Vx2 + 2.*Vx3 + Vx4) * self.ht / 6.
        Vy[:,:] += (Vy1 + 2.*Vy2 + 2.*Vy3 + Vy4) * self.ht / 6.
        
    
    
    def timestep(self, np.ndarray[np.float64_t, ndim=2] Bx,
                       np.ndarray[np.float64_t, ndim=2] By,
                       np.ndarray[np.float64_t, ndim=2] Vx,
                       np.ndarray[np.float64_t, ndim=2] Vy,
                       np.ndarray[np.float64_t, ndim=2] nBx,
                       np.ndarray[np.float64_t, ndim=2] nBy,
                       np.ndarray[np.float64_t, ndim=2] nVx,
                       np.ndarray[np.float64_t, ndim=2] nVy):
        
        cdef np.float64_t dxBx, dxBy, dyBx, dyBy
        cdef np.float64_t dxVx, dxVy, dyVx, dyVy
        
        cdef np.uint64_t ip, im, jp, jm
        cdef np.uint64_t i, j
        
        cdef np.float64_t hx = self.hx
        cdef np.float64_t hy = self.hy
        cdef np.uint64_t  nx = self.nx
        cdef np.uint64_t  ny = self.ny
        
        
        for i in range(0, nx):
            ip = (i+1) % nx
            im = (i-1) % nx
            
            for j in range(0, ny):
                jp = (j+1) % ny
                jm = (j-1) % ny
                
                dxBx = (Bx[ip,j] - Bx[im,j]) / hx
                dxBy = (By[ip,j] - By[im,j]) / hx
                dxVx = (Vx[ip,j] - Vx[im,j]) / hx
                dxVy = (Vy[ip,j] - Vy[im,j]) / hx
                
                dyBx = (Bx[i,jp] - Bx[i,jm]) / hy
                dyBy = (By[i,jp] - By[i,jm]) / hy
                dyVx = (Vx[i,jp] - Vx[i,jm]) / hy
                dyVy = (Vy[i,jp] - Vy[i,jm]) / hy
                
                
                ##############################
                # standard form
                
                # B_x
                nBx[i,j] = By[i,j] * dyVx \
                         + Vx[i,j] * dyBy \
                         - Bx[i,j] * dyVy \
                         - Vy[i,j] * dyBx
                    
                # B_y
                nBy[i,j] = Bx[i,j] * dxVy \
                         + Vy[i,j] * dxBx \
                         - By[i,j] * dxVx \
                         - Vx[i,j] * dxBy
                
                # V_x
                nVx[i,j] = By[i,j] * dyBx \
                         - By[i,j] * dxBy \
                         - Vx[i,j] * dxVx \
                         - Vy[i,j] * dyVx
                         
                # V_y
                nVy[i,j] = Bx[i,j] * dxBy \
                         - Bx[i,j] * dyBx \
                         - Vy[i,j] * dyVy \
                         - Vx[i,j] * dxVy
                
                
                ##############################
                # divergence form

#                # B_x
#                nBx[i,j] = By[i,j] * dyVx \
#                         + Vx[i,j] * dyBy \
#                         - Bx[i,j] * dyVy \
#                         - Vy[i,j] * dyBx
#                    
#                # B_y
#                nBy[i,j] = Bx[i,j] * dxVy \
#                         + Vy[i,j] * dxBx \
#                         - By[i,j] * dxVx \
#                         - Vx[i,j] * dxBy
#                
#                # V_x
#                nVx[i,j] = Bx[i,j] * dyBy \
#                         + By[i,j] * dyBx \
#                         - Vx[i,j] * dyVy \
#                         - Vy[i,j] * dyVx \
#                         - Vx[i,j] * dxVx \
#                         - Vx[i,j] * dxVx \
#                         + Bx[i,j] * dxBx \
#                         - By[i,j] * dxBy
#                    
#                # V_y
#                nVy[i,j] = Bx[i,j] * dxBy \
#                         + By[i,j] * dxBx \
#                         - Vx[i,j] * dxVy \
#                         - Vy[i,j] * dxVx \
#                         - Vy[i,j] * dyVy \
#                         - Vy[i,j] * dyVy \
#                         - Bx[i,j] * dyBx \
#                         + By[i,j] * dyBy
    
