#rotated mesh
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import transducers_ring
import matrix_rotation
from mpl_toolkits.mplot3d import Axes3D


class Rotated_mesh:
    def __init__(self, tr_i, z_middle, h_i, a_1, mesh_num):
        self.tr_i = tr_i            # transisters in i
        self.z_middle = z_middle    # midpoint of TinyLev
        self.h_i = h_i              # height of ring i  == z from transducers_ring1
        self.a_1 = a_1              # radius of spherical cap at ring 1 (will be measured)
        self.mesh_num = mesh_num    # number of points in a mesh representing one transducer
    
    def origin_mesh(self):
        trans = transducers_ring.Transducers(self.tr_i, 5, 0, self.mesh_num, 0)  #adjusting this cuts down time by a lot
        return trans

    # calculates the radius of the sphere based off known/measured quantities
    def radius_sphere(self):
        rSphere = (self.a_1**2 + self.h_i**2) / (2 * self.h_i)  #-> will be used to calculate actual values to replace h_i and a_i
        return rSphere
    
    # the angle that a transducer is rotated about the z-axis, determines it position on the ring
    # calculated based off known quantities
    def alpha(self):
        angle =[]
        for i in range(self.tr_i):
            angle.append(2 *np.pi / self.tr_i * (i))
        return angle
        
    # returns a GUESS for the a value for ring i
    def a_i(self):
        return (( 10 / (self.alpha()[1]))*1.5)    
  
    # the angle the transducers are rotated about the y-axis so that it faces the center once translated
    # calculated based off known quantities
    def theta(self):                                
        angle = np.arctan(self.a_i() / (self.z_middle - self.h_i))
        return angle
        
    # returns the mesh points of one transducer in a given ring before rotation
    # transducer_array = any 
    def unrotated_xyz_i(self, transducer_array, i):
        xyz = [transducer_array[0][i], transducer_array[1][i],
               transducer_array[2][i]]
        return xyz  
    
    def rotated_mesh(self):
        rotate = matrix_rotation.Rotation(1)
        #grabbing one ring at a time
        
        #rotating each transducers in ring x:
        rotated_array = []
        for i in range(self.tr_i):
            
            trans = self.origin_mesh()
            trans = trans.transducer()
            
            transducer_array = self.unrotated_xyz_i(trans, i)
            ry = rotate.rotation_y(transducer_array, -self.theta())   #blows up here! should do all rotations at 0, 0
            rx = rotate.rotation_z(ry, self.alpha()[i])
            #now translate
            rx[0] += self.a_i() * np.cos(self.alpha()[i]) #was aprox_a_i 
            rx[1] += self.a_i() * np.sin(self.alpha()[i]) 
            #and vertical translation
            rx[2] += self.h_i
            
            rotated_array.append(rx)
            
            
            
            
        return rotated_array
    
    
    #rotate by theta then alpha(radians to center) then translate
     
    #to get a better image, ensure that the radius of each ring is 
    #determined by the spherical cap
    
    
'''
data is stored as
[transducer i][0 = x, 1 = y, 2 = z]
'''    

### parameters for defining mesh grid

number_of_mesh_points = 10

# number of transducers in each ring
N = [ 6, 12, 18, 6, 12, 18]

# height of each ring [mm]
Z = [5, 15, 25, 195, 185, 175]


# array of coordinates for all transducers
Ntot = np.sum(N)



# TODO
# figure out a way to call the number of ring points in any given transducer ring
# (since, regardless of position, they all have the same size)
# use this number of points to create an array (p) of the xyz coordinates for all 72 transducers
# 
# this is a problem because one can only call the ring_points function an object of the class Transducers
# the only place there is such an object in this code is in the class Rotated_mesh
# the only place there is an object of this class is in the loop 

'''
transducers_ring.Transducers(input).ring_points()

p = np.zeros([Ntot,3,])
'''

# create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(-5, 210)



Rsphere = 0.

for i in range(len(N)):
    rot = Rotated_mesh(N[i], 100, Z[i] , 15., number_of_mesh_points)
    mesh = rot.rotated_mesh()

    if i == 0:
        Rsphere = rot.radius_sphere()
    
    for i in range(N[i]):
        x = mesh[i][0]
        y = mesh[i][1]
        z = mesh[i][2]
        ax.scatter(x, y, z)
#plt.show()


print(Rsphere)












print("rotated_mesh - Done.")