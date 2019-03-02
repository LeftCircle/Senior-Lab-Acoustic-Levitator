#Acoustic Levitation Test File
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
import rotated_mesh
import measurement_mesh
from mpl_toolkits.mplot3d import Axes3D


      
#initial condition constants
ntr = [6,12,18,6,12,18]
h_i = [5, 15, 25, 195, 185, 175]
radius_transducer = 5
meshN = 10
z_middle = 100
radius_largest_ring = 30 #guess

ntr_half = [6,12,18]
h_half = [5, 15, 25]


#initializing graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(0, 200)

#below method works, but graphs from the further functions are clean as
def graph():    
    #graphing transducer mesh
    for i in range(len(ntr)):
        tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i],
                                            meshN).rotated_mesh()
        for j in range(ntr[i]):
            x = tr_mesh[j][0]
            y = tr_mesh[j][1]
            z = tr_mesh[j][2]
            ax.scatter(x, y, z)
    
    #graphing measurement_mesh:
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(meshN), 0).m_mesh()
    for i in range(int(len(m_mesh[0][0]) / 2)):  #for some reason / 2 fixes it -> probably from how z was created in measurement mesh
        ax.scatter(m_mesh[0][i], m_mesh[1][i], m_mesh[2][i])
    plt.show()
        
   
#graph()

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

#quarter mesh for less computation. Explenation is in the matrix method slide

def quarter_mesh():
    x = [] ; y = [] ; z = []
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(1.5 * meshN), 1).m_mesh()

    for i in range(len(m_mesh[0][0])):
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    return np.concatenate((x)), np.concatenate(y), np.concatenate(z)

#quarter_mesh()        
#print(len(quarter_mesh()[0]))
#print(len(quarter_mesh()[1]))
#print(len(quarter_mesh()[2]))

xyz_t = transister_xyz_arrays()
ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2])
xyz = quarter_mesh()
ax.scatter(xyz[0], xyz[1], xyz[2])
py.show()








