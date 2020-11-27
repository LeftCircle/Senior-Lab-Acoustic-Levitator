#m_meshpoints
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
import rotated_mesh
from mpl_toolkits.mplot3d import Axes3D

# mesh of measurement points in between the transducer rings
'''
The half parameter in the below class is used to define what method of the 
internal mesh we are using. 
half = 0 : creates a 3D mesh over the entire measurement region (not useful)
half = 1 : creates a 2D mesh which is a slice of the 3D mesh at y = 0
           This method is used for visualizing the transducer mesh and the 
           measurement mesh
half = 2 : More efficient method than half = 1 for the same results
half = 3 : used for implementing the directionality of the transducers. This 
           creates an individual mesh for each transducer, and moves the mesh 
           in space as opposed to the transducers, accounting for the 
           directionality of the transducers. This must be used to calculate 
           the pressure at a given region
'''
class M_mesh:
    def __init__(self, radius_largest_ring, h_largest_ring, z_middle, m_mesh_n
                 , half, t_radius, shift=0):
        self.radius_largest_ring = radius_largest_ring
        self.m_mesh_n = m_mesh_n #meshpoints on one axis of rectangle defining mesh between top and bottom transducers
        self.h_largest_ring = h_largest_ring
        self.z_middle = z_middle
        self.half = half #1 for true, 0 for false
        self.t_radius = t_radius #radius of one transducer
        self.shift = shift
    def m_mesh(self):
        if self.half == 1:
            h_m_mesh_n = self.m_mesh_n
            x_start = -self.radius_largest_ring
            x_end   = 0 
            x = np.linspace(x_start, x_end, h_m_mesh_n)  #finer mesh b/c less points
            
            y = np.zeros(h_m_mesh_n)
            
            #x, y = np.meshgrid(x, y)
            #adding the radius+1 to ensure mesh does not collide with transducers
            z_start = self.h_largest_ring + (self.t_radius + 1)
            z_end   = 2 * self.z_middle - (self.t_radius + 1)
            z = np.linspace(z_start, z_end, h_m_mesh_n)
            
            # arrays of 3-D mesh points
            x_array = []
            y_array = [] 
            z_array = []
            for i in range(len(x)):
                x_array.append(x)
                y_array.append(y)
                z_array.append((np.ones(len(x))) * z[i])
                
            return x_array, y_array, z_array
        if self.half == 2: #potentially better method for m_mesh half
            h_m_mesh_n = self.m_mesh_n  
            
            x_end   = self.radius_largest_ring + self.t_radius
            x_start = -x_end
            x = np.linspace(x_start, x_end, h_m_mesh_n)  
            
            y = np.zeros(h_m_mesh_n)
            
            #x, y = np.meshgrid(x, y)
            
            z_start = self.h_largest_ring + (self.t_radius + 1)#plus 6 ensures not touching mesh transducers
            z_end   = 2 * self.z_middle - z_start
            z = np.linspace(z_start, z_end, h_m_mesh_n)
            
            x_array, z_array = np.meshgrid(x, z)
            y_array = np.ones([len(x_array[0]), len(x_array[0])]) * self.shift
            return x_array, y_array, z_array
        
        if self.half == 3: #if working with directionality (may break distance calculations)
            #creates a mesh centered at zero
            h_m_mesh_n = self.m_mesh_n  #half mesh mesh number
            
            x_end   = self.radius_largest_ring + self.t_radius
            x_start = -x_end
            x = np.linspace(x_start, x_end, h_m_mesh_n)  #finer mesh b/c less points
            
            y = np.zeros(h_m_mesh_n)
            
            #x, y = np.meshgrid(x, y)
            
            z_start = self.h_largest_ring #+ (self.t_radius + 1)#plus 6 ensures not touching mesh transducers
            z_end   = 2 * self.z_middle - z_start
            height = z_end - z_start
            z_start = -(height / 2) ; z_end = height / 2
            z = np.linspace(z_start, z_end, h_m_mesh_n)
            
            x_array, z_array = np.meshgrid(x, z)
            y_array = np.zeros([len(x_array[0]), len(x_array[0])])
            return x_array, y_array, z_array
        
        else:
            x_start = -self.radius_largest_ring
            x_end   =  self.radius_largest_ring
               
            y_start = x_start
            y_end   = x_end #b/c we are making a square base
            
            
            #mesh point created as a square
            x = np.linspace(x_start, x_end, self.m_mesh_n) 
            y = np.linspace(y_start, y_end, self.m_mesh_n)
            x, y = np.meshgrid(x, y) #temp values to get proper size
            
            #defining points inside the radius, and only keeping those
            # creates circular mesh
            r = np.sqrt(x**2 + y**2)
            inside = r < self.radius_largest_ring
            #making x, y, z same length
            x_inside = x[inside]
            y_inside = y[inside]
            
            # translating circular mesh up to create 3-D mesh
            z_start = self.h_largest_ring + 6 #plus 6 ensures not touching mesh transducers
            z_end   = 2 * self.z_middle - self.h_largest_ring - 6
            z = np.linspace(z_start, z_end, int(len(x_inside) / 2))  #-> decides how many points in z mesh
            
            # arrays of 3-D mesh points
            x_array = []
            y_array = [] 
            z_array = []
            for i in range(int(len(x_inside) / 2)):
                x_array.append(x_inside)
                y_array.append(y_inside)
                z_array.append(np.ones(len(x_inside)) * z[i])
            return x_array, y_array, z_array
 
'''
data is accessed by
[0 = x, 1 = y, 2 = z][array of points]
The mesh is not necessarily uniform, but it is still a mesh
'''
print("measurement_mesh - Done.")
