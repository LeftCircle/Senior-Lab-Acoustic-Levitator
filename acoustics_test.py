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
#ntr = [6,12,18,6,12,18]                                
ntr = [18,12,6,6,12,18]       #for some reason this one works with directionality but not the other way around
#ntr = [1,0,0,1,2,3]
#h_i = [0,0,0,50,50,50]
h_i = [5, 15, 25, 195, 185, 175] #[mm]
t_radius = 5.                # radius of transducer [mm]
t_meshN = 10                   # number of points in side length of square that represents transducer
m_meshN = 5                   # number of points in side length of square that represents transducer
z_middle = 100
radius_largest_ring = 30    # radius of transducer ring [mm] guess

omega = 40.e3               # frequency of emitted sound [Hz]
c = 343.e3                  # wave propogation velocity [mm/s]
amplitude = 0.01            # displacement amplitude #also a guess
phase = 0.                  # excitation phase of displacement
dens = 1.225e-9             # density of propogation medium [kg/mm^3]
wavelength = (c/omega)*(1e3)# wavelength of emitted sounds [mm]     ### TODO find actual numbers for these 2


t_mesh = len(transducers_ring.Transducers(1, 5, t_meshN).ring_points()[0])


ntr_half = np.array([ntr[0],ntr[1],ntr[2]])
#h_half = [5, 15, 25]


#initializing graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(0, 200)


# Accessing the data:
#    - how to output specific x, y, z components from the mesh

def transducer_mesh_full():
    x = [] ; y = [] ; z = []
    for i in range(len(ntr)):
        
        tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                        radius_largest_ring, m_meshN, t_radius).rotated_mesh()
        for j in range(ntr[i]):
            x.append(tr_mesh[j][0])
            y.append(tr_mesh[j][1])
            z.append(tr_mesh[j][2])
        
    #return np.array([x,y,z])
    return np.concatenate(x), np.concatenate(y), np.concatenate(z)
 
#data is super easy to grab from ^^
'''
xyz = transducer_mesh_full()
print(len(xyz[0]))
print(len(xyz[1]))
print(len(xyz[2]))
ax.scatter(xyz[0], xyz[1], xyz[2])
#py.show()
'''
def measurement_mesh_full():
    x = [] ; y = [] ; z = []
    m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                     int(m_meshN), 0, t_radius).m_mesh()
    for i in range(int(len(m_mesh[0][0]) / 2)):
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    #return np.array([x,y,z])
    return np.concatenate(x), np.concatenate(y), np.concatenate(z)

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
        tr_mesh = rotated_mesh.Rotated_mesh(ntr_half[i], z_middle, h_i[i], t_meshN,
                                        radius_largest_ring, m_meshN, t_radius).rotated_mesh()
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
                                     int(m_meshN), 2, t_radius).m_mesh()

    for i in range(len(m_mesh[0][0])):
        
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    
    # return np.array([x,y,z])
    # return x,y,z
    return np.concatenate(x), np.concatenate(y), np.concatenate(z)


################################################################################
    #p matrix stuff if fucked at the moment for this section
'''
will be used for directionality. Must also have half_mesh_m ^^^^
'''
#graphing transducer mesh and middle mesh
def directional_m_mesh():
    xm_ar = [] ; ym_ar = [] ; zm_ar = [] #these store the values of arrays for calculating P | SUPER IMPORTANT
    
    for i in range(len(ntr)):
        
        #tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
        #                                    radius_largest_ring, m_meshN, t_radius).rotated_mesh()
        
        if i < int(len(ntr) / 2 - 1):
            rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[2], t_meshN,
                                            radius_largest_ring, m_meshN, t_radius).rotated_middle_func(0)
        if i > int(len(ntr) / 2 - 1):
            rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[2], t_meshN,
                                            radius_largest_ring, m_meshN, t_radius).rotated_middle_func(1)
       
        for j in range(ntr[i]):
            xm = rotated_middle[j][0]
            ym = rotated_middle[j][1]
            zm = rotated_middle[j][2]  
            xm_ar.append(xm) ; ym_ar.append(ym) ; zm_ar.append(zm)
            #ax.scatter(xm, ym, zm)
    return xm_ar, ym_ar, zm_ar
def directional_t_mesh():
    x_ar = []  ; y_ar = []  ; z_ar = []
    
    for i in range(len(ntr)):
    
        if i < int(len(ntr) / 2 - 1):
            xyz_t_lower = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                        radius_largest_ring, m_meshN, t_radius).unrotated_rings(0)
        if i > int(len(ntr) / 2 - 1):
            xyz_t_lower = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[-1], t_meshN,
                        radius_largest_ring, m_meshN, t_radius).unrotated_rings(1)        
        for j in range(ntr[i]):
            
            x = xyz_t_lower[j][0] #; x = np.concatenate((x , half_mesh_m()[0]))
            y = xyz_t_lower[j][1] #; y = np.concatenate((y , half_mesh_m()[1]))
            z = xyz_t_lower[j][2] #; z = np.concatenate((z , half_mesh_m()[2]))
            #ax.scatter(x, y, z)
            x_ar.append(x) ; y_ar.append(y) ; z_ar.append(z)
            
            #ax.scatter(xm, ym, zm)
    return x_ar, y_ar, z_ar
'''
NOTE!!! May have to raise the transducers to their appropriate height to maintain
proper r values... done by including h[i] in rotated_mesh code. This is almost
certainly necessary. But also maybe not
'''
'''
ACCESSING DATA:
translated_tm_mesh()[0,1,2 = transister x/y/z | 3,4,5 = middle xyz][transducer i]
transducers are hopefully indexed such that 0-35 are the bottom transducers and
36-71 are the top.
'''
#now calculating pressure matrix using moved interior matrices and transducers
#at the origin

xyz_t = directional_t_mesh()
xyz_m = directional_m_mesh()
ax.scatter(xyz_m[0][15], xyz_m[1][15], xyz_m[2][15])
py.show()

m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_mesh, t_radius,
                                     phase, dens, wavelength)
# calculate the transfer and excitation matrices
t_matrix_full = []
for i in range(sum(ntr)):
    transfer_matrix = m_meth.t_matrix(xyz_t[:][i], xyz_m[:][i])
    if i == 0:
        t_matrix_full = transfer_matrix
    else:
        t_matrix_full = [t_matrix_full + transfer_matrix for t_matrix_full,
                        transfer_matrix in zip(t_matrix_full, transfer_matrix)]
print(t_matrix_full)
        
#u_matrix = m_meth.u_matrix(xyz_t)

# use T and U to calculate the pressure matrix
#pressure_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)

# capture the real part
#p = np.real(pressure_matrix)
###############################################################################
'''
#xyz_t = half_mesh_t()
#ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2])
xyz_mm = half_mesh_m()
ax.scatter(xyz_mm[0], xyz_mm[1], xyz_mm[2])
py.show()
'''


#testing matrix method things using half mesh
#xyz_t = half_mesh_t()
xyz_m = half_mesh_m()
print(len(xyz_m[0]), 'half mesh x')
print(len(xyz_m[1]), 'half mesh y')
print(len(xyz_m[2]), 'half mesh z')

xyz_t = transducer_mesh_full()
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
#print(np.shape(p))

# Plot the pressure map
# Need Pressure array to now be 2D with x = x and y = z

#creating the pressure array used for graphing
xy_size = int(np.sqrt(len(xyz_m[0])))
graphing_array = np.reshape(p, (xy_size, xy_size))
print(graphing_array[0])

'''
_______________________________________________________________________________
NOTE: I am not sure how our current program is grabbing the data. 
I am assuming that it begins by calculating P at each M point at height h
_______________________________________________________________________________
'''



'''
fig = plt.figure()
ax = fig.gca(projection='3d')
tempz = np.zeros([xy_size, xy_size])
# Make data.


# Plot the surface.
surf = ax.plot_surface(graphing_array[0],graphing_array[1], tempz,   cmap=cm.coolwarm,
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
''' #THIS GRAPH WORKS
#these will be the x and z values of our M space...must be a mesh though
#so the space before concatenated!
# NOTE: ONLY WORKS FOR HALF MESH
X = np.reshape(xyz_m[0], (xy_size, xy_size)) #only works for half mesh
Z = np.reshape(xyz_m[2], (xy_size, xy_size))

print(np.shape(X), np.shape(Z), 'shapes')
fig = pyplot.figure(figsize=(11,7), dpi=100)
pyplot.contourf(X, Z, graphing_array, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Z, graphing_array, cmap=cm.viridis)
#pyplot.streamplot(X, Z, u, v)
pyplot.xlabel('X')
pyplot.ylabel('Z');


py.show()
'''



print("acoustic_test - Done.")