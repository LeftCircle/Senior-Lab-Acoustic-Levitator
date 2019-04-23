#Acoustic Levitation with one transducer and reflector
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
import tiny_lev_positions2 as tlp

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(0, 120)
'''
# reflector mesh is 5 times that of the transducer mesh, 
t_meshN = 10
r_meshN = 50
m_meshN = 50
r_radius = 50
t_radius = 10
height = 30
omega = 2*np.pi*40.e3               # frequency of emitted sound [Hz]
c = 343.e3                  # wave propogation velocity [mm/s]
amplitude = 1            # displacement amplitude #also a guess
amplitude2 = 1
phase = np.pi                  # excitation phase of displacement
phase2 = 0
dens = 1.225e-9             # density of propogation medium [kg/mm^3]
wavelength = (c/omega) # wavelength of emitted sounds [mm]


def ring_points(radius, meshN, height):
        centerx = 0      
        centery = 0       
                            
        x_start = centerx - radius
        x_end   = centerx + radius
           
        y_start = centery - radius
        y_end   = centery + radius
        #mesh point created as a square
        x = np.linspace(x_start, x_end, meshN) 
        y = np.linspace(y_start, y_end, meshN)
        
        x, y = np.meshgrid(x, y)
        
        #defining points inside the radius, and only keeping those
        r = np.sqrt((x - centerx)**2 + (y - centery)**2)
        inside = r < radius 
        z = np.ones(len(x[inside]))*height
        
        return x[inside], y[inside], z



tran = ring_points(t_radius, t_meshN, 50)
reflector  = ring_points(r_radius, r_meshN, -15)

def half_mesh_m():
    x = [] ; y = [] ; z = []

    m_mesh = measurement_mesh.M_mesh(r_radius, height, int(height / 2),
                                     int(m_meshN), 2, t_radius).m_mesh()

    for i in range(len(m_mesh[0][0])):
        
        x.append(m_mesh[0][i]) 
        y.append(m_mesh[1][i])
        z.append(m_mesh[2][i])
    
    # return np.array([x,y,z])
    # return x,y,z
    return np.concatenate(x), np.concatenate(y), np.concatenate(z)

m_mesh = half_mesh_m()
#we have our meshes, so now matrix method!

m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_meshN, t_radius,
                                     phase, dens, wavelength)

p = np.zeros((len(m_mesh[0]),1), dtype = complex)


t_points = ([tran[0]] + [tran[1]] + [tran[2]])  
m_points = ([m_mesh[0]] + [m_mesh[1]] + [m_mesh[2]]) 
r_points = ([reflector[0]] + [reflector[1]] + [reflector[2]])  
#ax.scatter(tran[0], tran[1], tran[2])
#ax.scatter(m_points[0], m_points[1], m_points[2])
#ax.scatter(r_points[0], r_points[1], r_points[2])
py.show()

transfer_matrix = m_meth.t_matrix(t_points, m_points)
u_matrix = m_meth.u_matrix(t_points)
p_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)

p[:][:] += p_matrix[:][:]



transfer_matrix_rm = m_meth.t_matrix_rm(r_points, m_points)
transfer_matrix_tr = m_meth.t_matrix_tr(t_points, r_points)
t_matrix_rmtr = np.matmul(transfer_matrix_rm, 
                         transfer_matrix_tr)
p_matrix = 1.j / wavelength * m_meth.p_matrix(t_matrix_rmtr,
                                              u_matrix) 

p[:][:] += p_matrix[:][:]
'''
#TM RT TR U additon
transfer_matrix_rt = m_meth.t_matrix_rt(r_points, t_points)
t_matrix_tmrt = np.matmul(transfer_matrix, transfer_matrix_rt)
t_matrix_tmrttr = np.matmul(t_matrix_tmrt, transfer_matrix_tr)
p_matrix = (1.j / wavelength)**2 * m_meth.p_matrix(t_matrix_tmrttr, 
           u_matrix)

p[:][:] += p_matrix[:][:]

#third addition from matrix method paper
t_m_rmtr = np.matmul(transfer_matrix_rm, transfer_matrix_tr)
t_m_rmtrrt = np.matmul(t_m_rmtr, transfer_matrix_rt)
t_m_rmtrrttr = np.matmul(t_m_rmtrrt, transfer_matrix_tr)
p_matrix = (1.j/ wavelength)**3 * m_meth.p_matrix(t_m_rmtrrttr,
           u_matrix)
p[:][:] += p_matrix[:][:]

#fourth addition
t_m_tmrt = np.matmul(transfer_matrix, transfer_matrix_rt)
t_m_tmrttr = np.matmul(t_m_tmrt, transfer_matrix_tr)
t_m_tmrttrrt = np.matmul(t_m_tmrttr, transfer_matrix_rt)
t_m_tmrttrrttr = np.matmul(t_m_tmrttrrt, transfer_matrix_tr)
p_matrix = (1.j/ wavelength)**4 * m_meth.p_matrix(t_m_tmrttrrttr, 
           u_matrix)
   
p[:][:] += p_matrix[:][:]

'''
p = np.absolute(p)
py.show()



xyz_m = half_mesh_m()

# Plot the pressure map
# Need Pressure array to now be 2D with x = x and y = z

#creating the pressure array used for graphing
xy_size = int(np.sqrt(len(m_mesh[0])))
graphing_array = np.reshape(p, (xy_size, xy_size))
print(np.shape(graphing_array), 'shape of graphing array')
#print(graphing_array)

#these will be the x and z values of our M space...must be a mesh though
#so the space before concatenated!
# NOTE: ONLY WORKS FOR HALF MESH
X = np.reshape(xyz_m[0], (xy_size, xy_size)) #only works for half mesh
Z = np.reshape(xyz_m[2], (xy_size, xy_size))

print(graphing_array)

#print(np.shape(X), np.shape(Z), 'shapes')
#plt.hist2d(graphing_array[0],graphing_array[1], bins = (50,50), cmap = plt.cm.Reds)
#plt.show()


fig = pyplot.figure(figsize=(11,7), dpi=100)
plt.pcolormesh(X, Z, graphing_array, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
#pyplot.streamplot(X, Z, u, v)
pyplot.xlabel('X')
pyplot.ylabel('Z');
