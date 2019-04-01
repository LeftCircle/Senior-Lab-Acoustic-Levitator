#m_meshpoints
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
import rotated_mesh
from mpl_toolkits.mplot3d import Axes3D

# mesh of measurement points in between the transducer rings
class M_mesh:
    def __init__(self, radius_largest_ring, h_largest_ring, z_middle, m_mesh_n
                 , half, t_radius):
        self.radius_largest_ring = radius_largest_ring
        self.m_mesh_n = m_mesh_n #meshpoints on one axis of rectangle defining mesh between top and bottom transducers
        self.h_largest_ring = h_largest_ring
        self.z_middle = z_middle
        self.half = half #1 for true, 0 for false
        self.t_radius = t_radius #radius of one transducer
    def m_mesh(self):
        if self.half == 1:
            h_m_mesh_n = 2 * self.m_mesh_n
            x_start = -self.radius_largest_ring
            x_end   = 0 #might want to do -0.01 but who knows
            x = np.linspace(x_start, x_end, h_m_mesh_n)  #finer mesh b/c less points
            
            y = np.zeros(h_m_mesh_n)
            
            #x, y = np.meshgrid(x, y)
            
            z_start = self.h_largest_ring + (self.t_radius + 1)#plus 6 ensures not touching mesh transducers
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
            h_m_mesh_n = 2 * self.m_mesh_n  #half mesh mesh number
            
            x_end   = self.radius_largest_ring + self.t_radius
            x_start = -x_end
            x = np.linspace(x_start, x_end, h_m_mesh_n)  #finer mesh b/c less points
            
            y = np.zeros(h_m_mesh_n)
            
            #x, y = np.meshgrid(x, y)
            
            z_start = self.h_largest_ring + (self.t_radius + 1)#plus 6 ensures not touching mesh transducers
            z_end   = 2 * self.z_middle - z_start#(self.t_radius + 1)
            z = np.linspace(z_start, z_end, h_m_mesh_n)
            
            x_array, z_array = np.meshgrid(x, z)
            y_array = np.zeros([len(x_array[0]), len(x_array[0])])
            return x_array, y_array, z_array
        
        if self.half == 3: #if working with directionality (may break distance calculations)
            #creates a mesh centered at zero
            h_m_mesh_n = 2 * self.m_mesh_n  #half mesh mesh number
            
            x_end   = self.radius_largest_ring + self.t_radius
            x_start = -x_end
            x = np.linspace(x_start, x_end, h_m_mesh_n)  #finer mesh b/c less points
            
            y = np.zeros(h_m_mesh_n)
            
            #x, y = np.meshgrid(x, y)
            
            z_start = self.h_largest_ring + (self.t_radius + 1)#plus 6 ensures not touching mesh transducers
            z_end   = 2 * self.z_middle - z_start#(self.t_radius + 1)
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
'''
#testing
#array sizes need to be the same
mesh = M_mesh(10, 10, 50, 20)
m = mesh.m_mesh()
#print(m)  #([x,y,z][len(x,y,z)][len(x,y,z)])
print(len(m[1][0]))
print(len(m[2][0]))
# create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(-5, 210)
for i in range(len(m[1][0])):
    ax.scatter(m[0][i], m[1][i], m[2][i])
N = [ 6, 12, 18, 6, 12, 18]
Z = [5, 15, 25, 195, 185, 175]
for i in range(len(N)):
    rot = rotated_mesh.Rotated_mesh(N[i], 100, Z[i] , 15., 10)
    mesh = rot.rotated_mesh()
    if i == 0:
        Rsphere = rot.radius_sphere()
    
    for i in range(N[i]):
        x = mesh[i][0]
        y = mesh[i][1]
        z = mesh[i][2]
        ax.scatter(x, y, z)
plt.show() 
'''




