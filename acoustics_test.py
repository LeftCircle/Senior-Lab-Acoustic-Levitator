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
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

#   angular frequency, wave propogation velocity, displacement amplitude, 
#   number of points in transducer mesh, radius of transducer, 
#   excitation phase of displacement, density of propogating medium, 
#   wavelength of emitted sound
      
#initial condition constants
ntr = [6,12,18,6,12,18]                                
h_i = [5, 15, 25, 195, 185, 175] #[mm]
t_radius = 5.                # radius of transducer [mm]
meshN = 5                   # number of points in side length of square that represents transducer
z_middle = 100
radius_largest_ring = 30    # radius of transducer ring [mm] guess

omega = 1000.               # angular frequency #also a guess
c = 343.e3                  # wave propogation velocity [mm/s]
amplitude = 0.01            # displacement amplitude #also a guess
phase = 0.                  # excitation phase of displacement
dens = 1.225e-9             # density of propogation medium [kg/mm^3]
wavelength = 0.0002         # wavelength of emitted sounds [mm]     ### TODO find actual numbers for these 2


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

def transducer_mesh_full():
    x = [] ; y = [] ; z = []
    for i in range(len(ntr)):
        tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i],
                                            meshN).rotated_mesh()
        for j in range(ntr[i]):
            x.append(tr_mesh[j][0])
            y.append(tr_mesh[j][1])
            z.append(tr_mesh[j][2])
        
    return np.array([x,y,z])
    # return np.concatenate(x), np.concatenate(y), np.concatenate(z)
 
#data is super easy to grab from ^^
'''
xyz = transducer_mesh_full()
print(len(xyz[0]))
print(len(xyz[1]))
print(len(xyz[2]))
ax.scatter(xyz[0], xyz[1], xyz[2])
py.show()
'''
def measurement_mesh_full():
    x = [] ; y = [] ; z = []
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(1.5 * meshN), 0, t_radius).m_mesh()
    for i in range(int(len(m_mesh[0][0]) / 2)):
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    return np.array([x,y,z])
    # return np.concatenate(x), np.concatenate(y), np.concatenate(z)

''' 
data is super easy to grab from ^^
xyz = measurement_mesh_full()
ax.scatter(xyz[0], xyz[1], xyz[2])
py.show()
'''
'''
#simple combined graph
xyz_t = transducer_mesh_full()
ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2])
xyz_m = measurement_mesh_full()
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
            
    # return np.array([x,y,z])
    # return x,y,z
    return np.concatenate(x), np.concatenate(y), np.concatenate(z)

def half_mesh_m():
    x = [] ; y = [] ; z = []
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(1.5 * meshN), 1, t_radius).m_mesh()

    for i in range(len(m_mesh[0][0])):
        if i == 1:
            print(len(m_mesh[0][1]))
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    
    # return np.array([x,y,z])
    # return x,y,z
    return np.concatenate(x), np.concatenate(y), np.concatenate(z)

half_mesh_m()        
print(len(half_mesh_m()[0]))
print(len(half_mesh_m()[1]))
print(len(half_mesh_m()[2]))
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


#xyz_t = transducer_mesh_full()
#xyz_m = measurement_mesh_full()

# params for Matrix_method:
#   frequency, wave propogation velocity, displacement amplitude, 
#   number of points in transducer mesh, radius of transducer, 
#   excitation phase of displacement, density of propogating medium, 
#   wavelength of emitted sound
m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_mesh, t_radius,
                                     phase, dens, wavelength)


# calculate the transfer and excitation matrices
transfer_matrix = m_meth.t_matrix(xyz_t, xyz_m)
u_matrix = m_meth.u_matrix(xyz_t)

# use T and U to calculate the pressure matrix
pressure_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)

# capture the real part
p = np.real(pressure_matrix)


# output for debugging
# print("Transfer matrix:")
# print(np.shape(transfer_matrix))
# print("\nExcitation/displacement matrix:")
# print(np.shape(u_matrix))
print("\nPressure matrix:")
print(np.shape(pressure_matrix))
#print(pressure_matrix)
print("\nReal Part")
#print(p)

# Plot the pressure map

'''
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = xyz_m[0]
Y = xyz_m[1]

X, Y = np.meshgrid(X, Y)

Z = p

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''


'''
fig = pyplot.figure(figsize=(11, 7), dpi=100)

[X,Y,Z]  = np.meshgrid(100,100,100)


plt.surf(X,Y,Z,pressure_matrix)

py.show()
'''
'''
X = xyz_m[0]
Y = xyz_m[1]

X, Y = np.meshgrid(X, Y)


pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, p, cmap=cm.viridis)
pyplot.streamplot(X, Y, u, v)
pyplot.xlabel('X')
pyplot.ylabel('Y');


py.show()
'''



print("acoustic_test - Done.")