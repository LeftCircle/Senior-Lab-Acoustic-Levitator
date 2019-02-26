#rotated mesh
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
from mpl_toolkits.mplot3d import Axes3D


class Rotated_mesh:
    def __init__(self, tr_i, z_middle, h_i, a_i): 
        self.tr_i = tr_i #transisters in i
        self.z_middle = z_middle #midpoint of TinyLev
        self.h_i = h_i #height of ring i  == z from transducers_ring1
        self.a_i = a_i #radius of spherical cap at ring i
        
    def radius_sphere(self):
        rSphere = (self.a1**2 + self.h1**2) / (2 * self.h1)
        return rSphere
       
    def theta(self):                                #probably being calculated wrong
        angle = np.arctan(self.a_i / self.z_middle)
        return angle
        
    
    def unrotated_xyz_i(self, transducer_array, i):
        xyz = [transducer_array[0][i], transducer_array[1][i],
               transducer_array[2][i]]
        return xyz
    def aprox_a_i(self):
        return (2. * 5 / (2 * np.pi / (self.tr_i)))*1.5
    
    
    def rotated_mesh(self):
        rotate = matrix_rotation.Rotation(1)
        #grabbing one ring at a time
        trans = transducers_ring.Transducers(self.tr_i, 5, self.h_i, 25, 0)  #adjusting this cuts down time by a lot
        trans = trans.transducer()
        
        #rotating each transducers in ring x:
        rotated_array = []
        for i in range(self.tr_i):
            #rotate everything by pi/4 as test
            alpha = 2 *np.pi / self.tr_i * (i) ########might not be plus 1
            transducer_array = self.unrotated_xyz_i(trans, i)
            ry = rotate.rotation_y(transducer_array, -self.theta())
            rx = rotate.rotation_z(ry, alpha)
            #now translate
            rx[0] += self.aprox_a_i() * np.cos(alpha)
            rx[1] += self.aprox_a_i() * np.sin(alpha)
            
            rotated_array.append(rx)
            
            
            
            
        return rotated_array
    
    
    #rotate by theta then alpha(radians to center) then translate
        
'''
data is stored as
[transducer i][0 = x, 1 = y, 2 = z]
'''    
#checking
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-30, 30)
ax.set_ylim3d(-30, 30)
ax.set_zlim3d(-30, 30)
N = [6,12,18]
Z = [1 , 5, 10]
for n in N:
    rot = Rotated_mesh(n, 50, n / 3 , 10)
    rot = rot.rotated_mesh()

    for i in range(n):
        x = rot[i][0]
        y = rot[i][1]
        z = rot[i][2]
        ax.scatter(x, y, z)
plt.show()        
         
      