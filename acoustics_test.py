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

#initializing graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(0, 200)

def graph():    
    #graphing transducer mesh
    for i in range(len(ntr)):
        tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i],
                                            meshN).rotated_mesh()
        if i == 0:
            print(len(tr_mesh[0][0]))
            print(len(tr_mesh[0][1]))
            print(len(tr_mesh[0][2]))
        for i in range(ntr[i]):
            x = tr_mesh[i][0]
            y = tr_mesh[i][1]
            z = tr_mesh[i][2]
            ax.scatter(x, y, z)
    
    #graphing measurement_mesh:
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(meshN)).m_mesh()
    
    print(m_mesh[0][0][0])
    print(len(m_mesh[1][0]))
    print(len(m_mesh[2][0]))
    for i in range(int(len(m_mesh[0][0]) / 2)):  #for some reason / 2 fixes it -> probably from how z was created in measurement mesh
        ax.scatter(m_mesh[0][i], m_mesh[1][i], m_mesh[2][i])
    plt.show()
        
   
graph()




