#Acoustic Levitation main
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
      
#initial condition constants
ntr = [6,12,18,6,12,18]                                


h_i = [0, 10, 20, 200, 190, 180]
t_radius = 4.5                # radius of transducer [mm]
t_meshN = 10                   # number of points in side length of square that represents transducer
m_meshN = 30                   # number of points in side length of square that represents transducer
z_middle = 100
radius_largest_ring = 30    # radius of transducer ring [mm] guess

omega = 40.e3               # frequency of emitted sound [Hz]
c = 343.e3                  # wave propogation velocity [mm/s]
amplitude = 0.01            # displacement amplitude #also a guess
phase = 0.                  # excitation phase of displacement
dens = 1.225e-9             # density of propogation medium [kg/mm^3]
wavelength = (c/omega)*(1e3)# wavelength of emitted sounds [mm]     ### TODO find actual numbers for these 2

t_mesh = len(transducers_ring.Transducers(1, t_radius, t_meshN).ring_points()[0])


ntr_half = np.array([ntr[0],ntr[1],ntr[2]])
#h_half = [5, 15, 25]


#initializing graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(0, 210)


# Accessing the data:
#    - how to output specific x, y, z components from the mesh

def transducer_mesh_full():
    x = [] ; y = [] ; z = []
    for i in range(len(ntr)):
        
        tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                        radius_largest_ring, m_meshN, t_radius, h_i[0], h_i[3]).rotated_mesh()
        for j in range(ntr[i]):
            x.append(tr_mesh[j][0])
            y.append(tr_mesh[j][1])
            z.append(tr_mesh[j][2])
        #for k in range(int(ntr[i] / 1)):
            #ax.scatter(tr_mesh[k][0], tr_mesh[k][1], tr_mesh[k][2])
    #return np.array([x,y,z])
    return np.concatenate(x), np.concatenate(y), np.concatenate(z)
 
#data is super easy to grab from ^^

xyz = transducer_mesh_full()
print(len(xyz[0]))
print(len(xyz[1]))
print(len(xyz[2]))
#ax.scatter(xyz[0], xyz[1], xyz[2])
#py.show()

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


#half mesh for less computation. Explenation is in the matrix method slide
#I don't believe this is used
def half_mesh_t():
    x = [] ; y = [] ; z = []
    for i in range(len(ntr_half)):
        tr_mesh = rotated_mesh.Rotated_mesh(ntr_half[i], z_middle, h_i[i], t_meshN,
                                        radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3]).rotated_mesh()
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


#simple combined graph
xyz_t = transducer_mesh_full()
ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2])
xyz_m = half_mesh_m()
ax.scatter(xyz_m[0], xyz_m[1], xyz_m[2])
py.show()

################################################################################
'''
will be used for directionality. Must also have half_mesh_m ^^^^
'''
#graphing transducer mesh and middle mesh
def directional_m_mesh():
    xm_ar = [] ; ym_ar = [] ; zm_ar = [] #these store the values of arrays for calculating P | SUPER IMPORTANT
    
    for i in range(len(ntr)):
                
        if i <= int(len(ntr) / 2 - 1):
            rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                            radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3]).rotated_middle_func(0)
        else:
            rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                            radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3]).rotated_middle_func(1)
       
        for j in range(ntr[i]):
            xm = rotated_middle[j][0]
            ym = rotated_middle[j][1]
            zm = rotated_middle[j][2]  
            xm_ar.append(xm) ; ym_ar.append(ym) ; zm_ar.append(zm)
    return xm_ar, ym_ar, zm_ar



def directional_t_mesh():
    x_ar = []  ; y_ar = []  ; z_ar = []
    
    for i in range(len(ntr)):
    
        if i <= int(len(ntr) / 2 - 1):
            xyz_t = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                        radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3]).unrotated_rings(0)
        else:
            xyz_t = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                        radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3]).unrotated_rings(1)        
        for j in range(ntr[i]):
            
            x = xyz_t[j][0] #; x = np.concatenate((x , half_mesh_m()[0]))
            y = xyz_t[j][1] #; y = np.concatenate((y , half_mesh_m()[1]))
            z = xyz_t[j][2] #; z = np.concatenate((z , half_mesh_m()[2]))
            
            x_ar.append(x) ; y_ar.append(y) ; z_ar.append(z)
            
            #ax.scatter(x, y, z)
    return x_ar, y_ar, z_ar

'''
ACCESSING DATA: -- pre concatenate --
translated_tm_mesh()[0,1,2 = transister x/y/z | 3,4,5 = middle xyz][transducer i]
'''
#now calculating pressure matrix using moved interior matrices and transducers
#at the origin

xyz_t = directional_t_mesh()
xyz_m = directional_m_mesh()
#ax.scatter(xyz_m[0][0:4], xyz_m[1][0:4], xyz_m[2][0:4])
'''
NOTE: because the matix multiplication is the same for transducers across the 
origin, we are just oging to calculate for postive transducers then flip the 
pressure matrix and add this to the pressure matrix.

'''

m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_mesh, t_radius,
                                     phase, dens, wavelength)

#calculate the transfer and excitation matrices

p = np.zeros((len(xyz_m[0][0]),1))
#p_flipped = np.zeros(len(p))
#print(len(xyz_m[0]), 'xyz_m[0]', len(xyz_m[0][0]), '[0][0]')
#for i in range(sum(ntr)):


#this pressure calculation has all transducers emit from bottom left corner
for i in range(sum(ntr)): 
    
       
    t_points = ([xyz_t[0][i]] + [xyz_t[1][i]] + [xyz_t[2][i]])  
    m_points = ([xyz_m[0][i]] + [xyz_m[1][i]] + [xyz_m[2][i]]) 
    ax.scatter(xyz_t[0][i], xyz_t[1][i], xyz_t[2][i])
    ax.scatter(m_points[0], m_points[1], m_points[2])
    
    # NOTE: before we were concatenating all of the x/y/z to separate arrays
    transfer_matrix = m_meth.t_matrix(t_points, m_points)
    u_matrix = m_meth.u_matrix(t_points)
    p_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)
    
    #for some reason the p_matrix is the same regardless of orientation of 
    #the m_mesh. Could be because of matrix multiplication
    #to fix just rotate every other? array by 90 degrees?
    p_r = np.real(p_matrix)
    p[:][:] += p_r[:][:]
print(np.shape(p), 'shape of p')
#print(p)

py.show()
'''
#Functional pressure matrix with p_flip!
for k in range(int(len(ntr) / 1)):
    if k == 0:
        i = 0
    else:
        i = 0
        for l in range(k):
            i += ntr[k - l - 1]
            
    for j in range(int(ntr[k] / 2)):
        t_points = ([xyz_t[0][i]] + [xyz_t[1][i]] + [xyz_t[2][i]])  
        m_points = ([xyz_m[0][i]] + [xyz_m[1][i]] + [xyz_m[2][i]]) 
        #ax.scatter(xyz_t[0][i], xyz_t[1][i], xyz_t[2][i])
        #ax.scatter(m_points[0], m_points[1], m_points[2])
        
        transfer_matrix = m_meth.t_matrix(t_points, m_points)
        u_matrix = m_meth.u_matrix(t_points)
        p_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)
        
        p_r = np.real(p_matrix)
        
        p[:][:] += p_r[:][:] 
        
        i += 1
'''
'''
for k in range(int(len(ntr) / 2), len(ntr)):
    if k == 0:
        i = 0
    else:
        i = 0
        for l in range(k):
            i += ntr[k - 1]
    for j in range(int(ntr[k] / 2)):
        
        t_points = ([xyz_t[0][i]] + [xyz_t[1][i]] + [xyz_t[2][i]])  
        m_points = ([xyz_m[0][i]] + [xyz_m[1][i]] + [xyz_m[2][i]]) 
        #ax.scatter(xyz_t[0][i], xyz_t[1][i], xyz_t[2][i])
        #ax.scatter(m_points[0], m_points[1], m_points[2])
        
        transfer_matrix = m_meth.t_matrix(t_points, m_points)
        u_matrix = m_meth.u_matrix(t_points)
        p_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)
        
        p_r = np.real(p_matrix)
        
        p_top[:][:] += p_r[:][:] 
        
        i += 1     
'''
###############################################################################

#xyz_t = half_mesh_t()
#ax.scatter(xyz_t[0], xyz_t[1], xyz_t[2])
#xyz_mm = half_mesh_m()
#ax.scatter(xyz_mm[0][0:10], xyz_mm[1][0:10], xyz_mm[2][0:10])
#py.show()



#testing matrix method things using half mesh
#xyz_t = half_mesh_t()
xyz_m = half_mesh_m()
#print(len(xyz_m[0]), 'half mesh x')
#print(len(xyz_m[1]), 'half mesh y')
#print(len(xyz_m[2]), 'half mesh z')
#print(np.shape(xyz_m))

#xyz_t = transducer_mesh_full()
#xyz_m = measurement_mesh_full()

# params for Matrix_method:
#   frequency, wave propogation velocity, displacement amplitude, 
#   number of points in transducer mesh, radius of transducer, 
#   excitation phase of displacement, density of propogating medium, 
#   wavelength of emitted sound
'''
m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_mesh, t_radius,
                                     phase, dens, wavelength)
# calculate the transfer and excitation matrices
transfer_matrix = m_meth.t_matrix(xyz_t, xyz_m)
u_matrix = m_meth.u_matrix(xyz_t)
# use T and U to calculate the pressure matrix
pressure_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)
# capture the real part
p = np.real(pressure_matrix)
'''

# Plot the pressure map
# Need Pressure array to now be 2D with x = x and y = z

#creating the pressure array used for graphing
xy_size = int(np.sqrt(len(xyz_m[0])))
graphing_array = np.reshape(p, (xy_size, xy_size))
print(np.shape(graphing_array), 'shape of graphing array')
#print(graphing_array)
p_flipped = np.empty_like(graphing_array)
#print(p_flipped)
for i in range(len(graphing_array[0])):
    for j in range(int(len(graphing_array[0]) / 2)):
        p_flipped[i][j] = graphing_array[i][int(len(graphing_array) - 1 - j)]
        p_flipped[i][int(len(graphing_array) - 1 - j)] = graphing_array[i][j]

graphing_array = [graphing_array + p_flipped for graphing_array,
                  p_flipped in zip(graphing_array, p_flipped)]   
'''
graphing_array_bottom = np.reshape(p_bottom, (xy_size, xy_size))
print(np.shape(graphing_array_bottom), 'shape of graphing array')
#print(graphing_array)
p_flipped = np.empty_like(graphing_array_bottom)
#print(p_flipped)
for i in range(len(graphing_array_bottom[0])):
    for j in range(int(len(graphing_array_bottom[0]) / 2)):
        p_flipped[i][j] = graphing_array_bottom[i][int(len(graphing_array_bottom) - 1 - j)]
        p_flipped[i][int(len(graphing_array_bottom) - 1 - j)] = graphing_array_bottom[i][j]

graphing_array_bottom = [graphing_array_bottom + p_flipped for graphing_array_bottom,
                  p_flipped in zip(graphing_array_bottom, p_flipped)]        
'''

#graphing_array = [graphing_array + graphing_array_bottom for graphing_array, 
#                  graphing_array_bottom in zip(graphing_array , graphing_array_bottom)]     
#print(np.shape(graphing_array), 'graphing array!')
#print(len(graphing_array[0]))

#print(graphing_array[0])



 #THIS GRAPH WORKS
#these will be the x and z values of our M space...must be a mesh though
#so the space before concatenated!
# NOTE: ONLY WORKS FOR HALF MESH
X = np.reshape(xyz_m[0], (xy_size, xy_size)) #only works for half mesh
Z = np.reshape(xyz_m[2], (xy_size, xy_size))

#print(np.shape(X), np.shape(Z), 'shapes')
fig = pyplot.figure(figsize=(11,7), dpi=100)
pyplot.contourf(X, Z, graphing_array, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Z, graphing_array, cmap=cm.viridis)
#pyplot.streamplot(X, Z, u, v)
pyplot.xlabel('X')
pyplot.ylabel('Z');


#py.show()




print("acoustic_test - Done.")