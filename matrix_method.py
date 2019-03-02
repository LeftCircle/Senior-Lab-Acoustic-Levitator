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
    def __init__(self, omega, c, amplitude, t_meshN, t_radius):
        self.omega = omega #anglular freq???
        self.c = c
        self.amplitude = amplitude
        self.t_meshN = t_meshN
        self.t_radius = t_radius
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
    #notation is bad again. both n and t represent transisters
    def sn(self): #radius of transducer cell
        return(self.radius_t / self.t_meshN)**2
    
    
    def r_nm_m(self, t_points, m_points): #r_nm matrix
        m_length = len(m_points[0]) ; t_length = len(t_points[0])
        r_nm_matrix = np.zeros(m_length, t_length)
        for i in range(m_length):
            for j in range(t_length):
                r_nm_matrix[i,j] = np.sqrt((m_points[0][i]-t_points[0][j])**2 + 
                           (m_points[1][i]-t_points[1][j])**2 +
                           (m_points[2][i]-t_points[2][j])**2) 
        return r_nm_matrix
    
    def t_nm(self, r_nm): #t_nm^TM from matrix method paper
        k = self.omega / self.c
        # I think sqrt(-1) = 1j
        return self.sn() * (np.exp(-1j*k*r_nm)) / r_nm
    
    def u_n(self, t_point, excitation_phase_n):
        return self.amplitude * np.exp(1j * excitation_phase_n) 
        

    def t_matrix(self, t_points, m_points):
        m_length = len(m_points[0]) ; t_length = len(t_points[0])
        t_m = np.zeros(m_length, t_length) #m rows, t collumns
        r_nm = self.r_nm_m(t_points, m_points)
        for i in range(m_length): #may not be the most efficient method
            for j in range(t_length):
                t_m[i, j] = self.t_nm(r_nm[i][j]) 
                
            













