#matrix method:
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
import rotated_mesh
import measurement_mesh
from mpl_toolkits.mplot3d import Axes3D

class Matrix_method:
    # params:
    #   frequency, wave propogation velocity, displacement amplitude, 
    #   number of points in transducer mesh, radius of transducer, 
    #   excitation phase of displacement, density of propogating medium, 
    #   wavelength of emitted sound
    def __init__(self, omega, c, amplitude, t_meshN, t_radius, 
                 excitation_phase_n, density, wavelength):
        self.omega = omega #anglular freq???
        self.c = c
        self.amplitude = amplitude # amplitude of displacement
        self.t_meshN = t_meshN 
        self.t_radius = t_radius
        self.excitation_phase_n = excitation_phase_n # phase of displacement 
        self.density = density
        self.wavelength = wavelength
        
        # TODO: find out what amplitude and excitatin_phase_n actually are
        
    
    '''
    note - we don't have to calculate both the top force from both the top and
    the bottom. We actually just need to calculate the force from the bottom
    to all points in one 2D vertical slice. The slice only needs to go to the 
    center.
    If we add the inverse of the force from the bottom array, it is the same as
    simuilating the top.
    We can then rotate the result by 2pi about Z to get the 3D behavior due to 
    symmetry
    If we want to see one slice, then rotate result by pi about the z to show
    full behavior
    '''
    '''
    isntead, we could take a slice that goes halfway through some bisection 
    of the sphere, then rotate that slice by pi in order to show what is occur-
    ing at any slice in the TinyLev
    '''
    
    #notation is bad again. both n and t represent transisters
    def sn(self): #radius of transducer cell
        return(1 * self.t_radius / self.t_meshN)**2
    
    # calculates distance between transducer points (n,t) and measurement points (m)
    # t_points, m_points are 2-D arrays of x,y,z points
    def r_nm_m(self, t_points, m_points): #r_nm matrix
        m_length = len(m_points[0]) ; t_length = len(t_points[0])
        r_nm_matrix = np.zeros([m_length, t_length], dtype=complex) ###making complex array
                                                                    #hopefully with real values
        for i in range(m_length):
            for k in range(t_length):
                r_nm_matrix[i,k] = np.sqrt((m_points[0][i]-t_points[0][k])**2 + 
                           (m_points[1][i]-t_points[1][k])**2 +
                           (m_points[2][i]-t_points[2][k])**2) 
        return r_nm_matrix
    
    # element of transfer matrix between transducer points (n) and measurement points (m)
    def t_nm(self, r_nm): #t_nm^TM from matrix method paper
        k = self.omega / self.c
        # I think sqrt(-1) = 1.j
        return self.sn() * (np.exp(-1.j*k*r_nm)) / r_nm
    
    # calculates displacement of each cell s_n due to oscillation from sound waves
    def u_n(self):
        return self.amplitude * np.exp(1.j * self.excitation_phase_n)
        
    # transfer matrix between transducer points (n) and measurement points (m)
    def t_matrix(self, t_points, m_points):
        m_length = len(m_points[0]) ; t_length = len(t_points[0])
        
        #print(m_length, 'm_length', t_length, 't_length |', 't_matrix' )
        
        t_m = np.zeros([m_length, t_length], dtype=complex) #m rows, t collumns
        r_nm = self.r_nm_m(t_points, m_points)
        
        for i in range(m_length): #may not be the most efficient method
            for k in range(t_length): #don't use j bc imaginary numbers
                t_m[i, k] = self.t_nm(r_nm[i][k]) 
                
                #potentially make everything a complex number to account for it?
        return t_m
    
    # assembles displacement matrix
    def u_matrix(self, t_points):
        t_length = len(t_points[0])
        #print(t_length, 't_length', 'for u_matrix')
        u_mat = np.zeros([t_length, 1],  dtype = complex)
        for i in range(t_length):
            u_mat[i] = self.u_n()
        return u_mat
            
    # pressure matrix at measurement points (m)
    # here t_mat is the transfer matrix (not transducers)
    def p_matrix(self, t_mat, u_mat):
        
        p_prefactor = self.omega * self.density * self.c / self.wavelength
        
        # TEST 
        # print sizes of t and u matrices to ensure they can be multplied
        # print("Pressure Calculation")
        # print("T Matrix: " + str(np.shape(t_mat)))
        # print("U Matrix: " + str(np.shape(u_mat)))
        
        return p_prefactor * np.matmul(t_mat, u_mat) * 1.e-3 # (to convert the mm to m, giving the result in units of Pascals)
        
        
        

                
            













print("matrix_method - Done.")