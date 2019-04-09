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
    
    #notation here is bad again. both n and t represent transducers
    def sn(self): #radius of transducer cell
        return(1 * self.t_radius / self.t_meshN)**2
    
    # calculates distance between transducer points (n,t) and measurement points (m)
    # t_points, m_points are 2-D arrays of x,y,z points
    def r_nm_m(self, t_points, m_points): #r_nm matrix
        m_length = len(m_points[0]) ; t_length = len(t_points[0])
        r_nm_matrix = np.zeros([m_length, t_length], dtype=complex) ###making complex array
                                                                   
        for i in range(m_length):
            for k in range(t_length): #potentially change how this is indexed
                r_nm_matrix[i,k] = np.sqrt((m_points[0][i]-t_points[0][k])**2 + 
                           (m_points[1][i]-t_points[1][k])**2 +
                           (m_points[2][i]-t_points[2][k])**2)  
        return r_nm_matrix
    
    # element of transfer matrix between transducer points (n) and measurement points (m)
    def t_nm(self, r_nm): #t_nm^TM from matrix method paper
        k = self.omega / self.c
        return self.sn() * (np.exp(-1.j*k*r_nm)) / r_nm
    
    # calculates displacement of each cell s_n due to oscillation from sound waves
    def u_n(self):
        return self.amplitude * np.exp(1.j * self.excitation_phase_n)
        
    # transfer matrix between transducer points (n) and measurement points (m)
    def t_matrix(self, t_points, m_points):
        m_length = len(m_points[0]) ; t_length = len(t_points[0])
        t_m = np.zeros([m_length, t_length], dtype=complex) #m rows, t collumns
        r_nm = self.r_nm_m(t_points, m_points)
        
        
        for i in range(m_length): 
            for k in range(t_length): 
                t_m[i, k] = self.t_nm(r_nm[i][k]) 
                
        return t_m
    
    # assembles displacement matrix
    def u_matrix(self, t_points):
        t_length = len(t_points[0])
        u_mat = np.zeros([t_length, 1],  dtype = complex)
        for i in range(t_length):
            u_mat[i] = self.u_n()
        return u_mat
            
    # pressure matrix at measurement points (m)
    # here t_mat is the transfer matrix (not transducers)
    def p_matrix(self, t_mat, u_mat): 
        p_prefactor = self.omega * self.density * self.c / self.wavelength 
        return p_prefactor * np.matmul(t_mat, u_mat)# * 1.e-3 # (to convert the mm to m, giving the result in units of Pascals)
        

print("matrix_method - Done.")