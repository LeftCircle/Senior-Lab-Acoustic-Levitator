#Acoustic Levitation Test File
'''
This document shows how to access all data types, and is also home to a guess
for initial conditions for testing purposes, as well as the half mesh, 
which are certain conditions that could cut down on computational time. The
half mesh process is described in the matrix method document. 
_______________________________________________________________________________

eventually, this code will be implimented into a main() function where all of 
the code is executed. For now, this is a reference and testing environment.  
'''

import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
import rotated_mesh
import measurement_mesh
import matrix_method
from mpl_toolkits.mplot3d import Axes3D


      
#initial condition constants
ntr = [6,12,18,6,12,18]
h_i = [5, 15, 25, 195, 185, 175]
t_radius = 5
meshN = 5
z_middle = 100
radius_largest_ring = 30 #guess
omega = 1000 #also a guess
c = 273
amplitude = 0.01 #also a guess

t_mesh = len(transducers_ring.Transducers(1, 5, meshN).ring_points()[0])


ntr_half = [6,12,18]
#h_half = [5, 15, 25]


#initializing graph
def graph():
    g = False
    if g == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-100, 100)
        ax.set_ylim3d(-100, 100)
        ax.set_zlim3d(0, 200)
graph()
# Accessing the data:
#    - how to output specific x, y, z components from the mesh

def transister_xyz_arrays():
    x = [] ; y = [] ; z = []
    for i in range(len(ntr)):
        tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i],
                                            meshN).rotated_mesh()
        for j in range(ntr[i]):
            x.append(tr_mesh[j][0])
            y.append(tr_mesh[j][1])
            z.append(tr_mesh[j][2])
        
    return np.concatenate((x)), np.concatenate(y), np.concatenate(z)
 
#data is super easy to grab from ^^
'''
xyz = transister_xyz_arrays()
print(len(xyz[0]))
print(len(xyz[1]))
print(len(xyz[2]))
ax.scatter(xyz[0], xyz[1], xyz[2])
py.show()
'''
def measurement_xyz_arrays():
    x = [] ; y = [] ; z = []
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(1.5 * meshN), 0).m_mesh()
    for i in range(int(len(m_mesh[0][0]) / 2)):
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    return np.concatenate((x)), np.concatenate(y), np.concatenate(z)

''' 
data is super easy to grab from ^^
xyz = measurement_xyz_arrays()
ax.scatter(xyz[0], xyz[1], xyz[2])
py.show()
'''
'''
#simple combined graph
xyz_t = transister_xyz_arrays()
ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2])
xyz_m = measurement_xyz_arrays()
ax.scatter(xyz_m[0], xyz_m[1], xyz_m[2])
py.show()
'''

#half mesh for less computation. Explenation is in the matrix method slide
def half_mesh_t():
    x = [] ; y = [] ; z = []
    for i in range(len(ntr_half)):
        tr_mesh = rotated_mesh.Rotated_mesh(ntr_half[i], z_middle, h_i[i],
                                            meshN).rotated_mesh()
        for j in range(ntr[i]):
            x.append(tr_mesh[j][0])
            y.append(tr_mesh[j][1])
            z.append(tr_mesh[j][2])
        
    return np.concatenate((x)), np.concatenate(y), np.concatenate(z)

def half_mesh_m():
    x = [] ; y = [] ; z = []
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(1.5 * meshN), 1).m_mesh()

    for i in range(len(m_mesh[0][0])):
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    return np.concatenate((x)), np.concatenate(y), np.concatenate(z)

#half_mesh()        
#print(len(half_mesh()[0]))
#print(len(half_mesh()[1]))
#print(len(half_mesh()[2]))
'''
xyz_t = half_mesh_t()
ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2])
xyz = half_mesh_m()
ax.scatter(xyz[0], xyz[1], xyz[2])
py.show()
'''


#testing matrix method things using half mesh
xyz_t = half_mesh_t()
xyz_m = half_mesh_m()
m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_mesh, t_radius,
                                     np.pi / 4)
print(len(m_meth.t_matrix(xyz_t, xyz_m)[0]))
print(len(m_meth.u_matrix(xyz_t)))

#all seems to be a go







