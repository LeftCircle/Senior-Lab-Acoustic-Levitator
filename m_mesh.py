#m_meshpoints
#rotated mesh
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
import rotated_mesh
from mpl_toolkits.mplot3d import Axes3D

class M_mesh:
    def __init__(self, radius_largest_ring, h_largest_ring, z_middle, m_mesh_n):
        self.radius_largest_ring = radius_largest_ring
        self.m_mesh_n = m_mesh_n #meshpoints on one axis of rectangle defining mesh between top and bottom transducers
        self.h_largest_ring = h_largest_ring
        self.z_middle = z_middle
    def m_mesh(self):
        x_start = -self.radius_largest_ring
        x_end   =  self.radius_largest_ring
           
        y_start = x_start
        y_end   = x_end #b/c we are making a square base
        
        
        #mesh point created as a square
        x = np.linspace(x_start, x_end, self.m_mesh_n) 
        y = np.linspace(y_start, y_end, self.m_mesh_n)
        x, y = np.meshgrid(x, y) #temp values to get proper size
        
        #defining points inside the radius, and only keeping those
        r = np.sqrt(x**2 + y**2)
        inside = r < self.radius_largest_ring
        #making x, y, z same length
        x_inside = x[inside]
        y_inside = y[inside]
        
        z_start = self.h_largest_ring + 6 #plus 6 ensures not touching mesh transducers
        z_end   = 2 * self.z_middle - self.h_largest_ring - 6
        z = np.linspace(z_start, z_end, int(len(x_inside) / 2))
        
        x_array = []
        y_array = [] 
        z_array = []
        for i in range(int(len(x_inside) / 2)):
            x_array.append(x_inside)
            y_array.append(y_inside)
            z_array.append(np.ones(len(x_inside)) * z[i])
            
        
        
        
        
        
        
        ############# np.meshgrid may cause interesting indexing of xyz########
        return x_array, y_array, z_array
 
'''
data is accessed by
[0 = x, 1 = y, 2 = z][array of points]
The mesh is not necessarily uniform, but it is still a mesh
'''

#testing
#array sizes need to be the same
mesh = M_mesh(10, 10, 50, 20)
m = mesh.m_mesh()
#print(m)  #([x,y,z][len(x,y,z)][len(x,y,z)])
print(len(m[1][0]))
print(len(m[2][0]))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-40, 40)
ax.set_ylim3d(-40, 40)
ax.set_zlim3d(20, 100)




for i in range(len(m[1][0])):
    ax.scatter(m[0][i], m[1][i], m[2][i])

plt.show()    

    
        
        



