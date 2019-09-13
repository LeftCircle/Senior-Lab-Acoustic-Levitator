#Acoustic Levitation main
#does not include directionality
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
ntr_l = [6,12,18]
ntr_u = [6,12,18]
h_i = [0, 0.8, 1.6, 119.1, 118.3, 117.5]
h_l = [0, 0.8, 1.6] ; h_u = [119.1, 118.3, 117.5]
r_i = [10.52, 21.35, 30.62, 10.52, 21.35, 30.62]
r_l = [10.52, 21.35, 30.62]; r_u = r_l
#h_i = [0, 4, 8, 100, 96, 92]
#h_i = [0, 10, 20, 200, 190, 180]
#h_l = [0,10,20] ; h_u = [200, 190, 180]
t_radius = 4.5                # radius of transducer [mm]
t_meshN = 10                   # number of points in side length of square that represents transducer
m_meshN = 50 
z_middle = 59.55                  # number of points in side length of square that represents transducer
#z_middle = 100
radius_largest_ring = 30.62    # radius of transducer ring [mm] guess

omega = 2*np.pi*40.e3               # frequency of emitted sound [Hz]
c = 343.e3                  # wave propogation velocity [mm/s]
amplitude = 1            # displacement amplitude #also a guess
amplitude2 = 1
phase = np.pi                  # excitation phase of displacement
phase2 = 0
dens = 1.225e-9             # density of propogation medium [kg/mm^3]
wavelength = (c/omega)# wavelength of emitted sounds [mm]     ### TODO find actual numbers for these 2


t_mesh = len(transducers_ring.Transducers(1, t_radius, t_meshN).ring_points()[0])


ntr_half = np.array([ntr[0],ntr[1],ntr[2]])
#h_half = [5, 15, 25]


#initializing graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(0, 120)

def main():
    # Accessing the data:
    #    - how to output specific x, y, z components from the mesh
    
    def transducer_mesh_full():
        x = [] ; y = [] ; z = []
        for i in range(len(ntr)):
            
            tr_mesh = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                            radius_largest_ring, m_meshN, t_radius, h_i[0], h_i[3], r_i[i]).rotated_mesh()
            for j in range(ntr[i]):
                x.append(tr_mesh[j][0])
                y.append(tr_mesh[j][1])
                z.append(tr_mesh[j][2])
            #for k in range(int(ntr[i] / 1)):
                #ax.scatter(tr_mesh[k][0], tr_mesh[k][1], tr_mesh[k][2])
        #return np.array([x,y,z])
        return np.concatenate(x), np.concatenate(y), np.concatenate(z)
     
    #data is super easy to grab from ^^
    
    
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
    #data is super easy to grab from ^^
    xyz = transducer_mesh_full()
    ax.scatter(xyz[0], xyz[1], xyz[2])
    py.show()
    '''
    
    def half_mesh_t(lower_or_upper):
        if lower_or_upper == 0:
            x = [] ; y = [] ; z = []
            for i in range(len(ntr_l)):
                tr_mesh = rotated_mesh.Rotated_mesh(ntr_l[i], z_middle, h_l[i], t_meshN,
                                                radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).rotated_mesh()
                for j in range(ntr[i]):
                    x.append(tr_mesh[j][0])
                    y.append(tr_mesh[j][1])
                    z.append(tr_mesh[j][2])
        else:
            x = [] ; y = [] ; z = []
            for i in range(len(ntr_u)):
                tr_mesh = rotated_mesh.Rotated_mesh(ntr_u[i], z_middle, h_u[i], t_meshN,
                                                radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).rotated_mesh()
                for j in range(ntr[i]):
                    x.append(tr_mesh[j][0])
                    y.append(tr_mesh[j][1])
                    z.append(tr_mesh[j][2])
                
        # return np.array([x,y,z])
        # return x,y,z
        return np.concatenate(x), np.concatenate(y), np.concatenate(z)
    
    #actually gives a 2D slice of the 3D mesh
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
    
    '''
    will be used for directionality. Must also have half_mesh_m ^^^^
    '''
    #graphing transducer mesh and middle mesh
    def directional_m_mesh():
        xm_ar = [] ; ym_ar = [] ; zm_ar = [] #these store the values of arrays for calculating P | SUPER IMPORTANT
        
        for i in range(len(ntr)):
                    
            if i <= int(len(ntr) / 2 - 1):
                rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                                radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).rotated_middle_func(0)
            else:
                rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                                radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).rotated_middle_func(1)
           
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
        
            if i <= int(len(ntr) / 2 - 1):
                xyz_t = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                            radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).unrotated_rings(0)
            else:
                xyz_t = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                            radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).unrotated_rings(1)        
            for j in range(ntr[i]):
                
                x = xyz_t[j][0] #; x = np.concatenate((x , half_mesh_m()[0]))
                y = xyz_t[j][1] #; y = np.concatenate((y , half_mesh_m()[1]))
                z = xyz_t[j][2] #; z = np.concatenate((z , half_mesh_m()[2]))
                
                x_ar.append(x) ; y_ar.append(y) ; z_ar.append(z)
                
                #ax.scatter(x, y, z)
        return x_ar, y_ar, z_ar
    py.show()
    '''
    ACCESSING DATA: -- pre concatenate --
    translated_tm_mesh()[0,1,2 = transister x/y/z | 3,4,5 = middle xyz][transducer i]
    '''
    
    m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_mesh, t_radius,
                                         phase, dens, wavelength)
    m_meth2 = matrix_method.Matrix_method(omega, c, amplitude2, t_mesh, t_radius,
                                         phase2, dens, wavelength)
    
    m_meth_m = matrix_method.Matrix_method(omega, c, amplitude, t_mesh,
                                radius_largest_ring, phase, dens, wavelength)
    m_meth_m2 = matrix_method.Matrix_method(omega, c, amplitude2, t_mesh,
                                radius_largest_ring, phase2, dens, wavelength)
    #calculate the transfer and excitation matrices
    p = np.zeros((len(xyz_m[0]),1), dtype = complex)
    
    
    #matrix method like what is done in the paper, assuming that the top and
    #bottom caps are just one transducer
            
    t_pointsl = half_mesh_t(0)  
    t_pointsu = half_mesh_t(1)
    m_points = half_mesh_m() 
    #ax.scatter(xyz_t[0][i], xyz_t[1][i], xyz_t[2][i])
    #ax.scatter(m_points[0], m_points[1], m_points[2])
    
    #lower transfer matrices and pressure
    transfer_matrixl = m_meth.t_matrix(t_pointsl, m_points)
    u_matrixl = m_meth.u_matrix(t_pointsl)
    p_matrixl = m_meth.p_matrix(transfer_matrixl, u_matrixl)
    
    p[:][:] += p_matrixl[:][:]
    
    #t1 to t2, t2 to middle    
    transfer_matrix_rm = m_meth.t_matrix(t_pointsu, m_points)
    transfer_matrix_tt = m_meth.t_matrix(t_pointsl, t_pointsu)
    t_matrix_ttm = np.matmul(transfer_matrix_rm, 
                             transfer_matrix_tt)
    p_matrix = 1.j / wavelength * m_meth.p_matrix(t_matrix_ttm,
                                                  u_matrixl) 
    
    p[:][:] += p_matrix[:][:]
    
    #TM RT TR U additon
    transfer_matrix_rt = m_meth.t_matrix(t_pointsu, t_pointsl)
    t_matrix_tmrt = np.matmul(transfer_matrixl, transfer_matrix_rt)
    t_matrix_tmrttr = np.matmul(t_matrix_tmrt, transfer_matrix_tt)
    p_matrix = (1.j / wavelength)**2 * m_meth.p_matrix(t_matrix_tmrttr, 
               u_matrixl)
    
    p[:][:] += p_matrix[:][:]
    
    
    #upper transfer matrices and pressure
    transfer_matrixu = m_meth2.t_matrix(t_pointsu, m_points)
    u_matrixu = m_meth2.u_matrix(t_pointsu)
    p_matrixu = m_meth2.p_matrix(transfer_matrixu, u_matrixu)
    
    p[:][:] += p_matrixu[:][:]
    
    #t1 to t2, t2 to middle    
    transfer_matrix_rm = m_meth2.t_matrix(t_pointsl, m_points)
    transfer_matrix_tt = m_meth2.t_matrix(t_pointsu, t_pointsl)
    t_matrix_ttm = np.matmul(transfer_matrix_rm, 
                             transfer_matrix_tt)
    p_matrix = 1.j / wavelength * m_meth.p_matrix(t_matrix_ttm,
                                                  u_matrixu) 
    
    p[:][:] += p_matrix[:][:]
    
    #TM RT TR U additon
    transfer_matrix_rt = m_meth2.t_matrix(t_pointsl, t_pointsu)
    t_matrix_tmrt = np.matmul(transfer_matrixu, transfer_matrix_rt)
    t_matrix_tmrttr = np.matmul(t_matrix_tmrt, transfer_matrix_tt)
    p_matrix = (1.j / wavelength)**2 * m_meth.p_matrix(t_matrix_tmrttr, 
               u_matrixu)
    
    p[:][:] += p_matrix[:][:]
    
    
    #final pressure calculation
    p = np.absolute(p)
    py.show()
    
    
    
    xyz_m = half_mesh_m()
    
    # Plot the pressure map
    # Need Pressure array to now be 2D with x = x and y = z
    
    #creating the pressure array used for graphing
    xy_size = int(np.sqrt(len(xyz_m[0])))
    graphing_array = np.reshape(p, (xy_size, xy_size))
    print(np.shape(graphing_array), 'shape of graphing array')
    #print(graphing_array)
    
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
    
    #finding the center points used for graphing
    centerline = []
    midpoint = int(m_meshN / 2)
    height = np.linspace(h_l[-1], h_u[-1], m_meshN)
    for i in range(len(graphing_array[0])):
        centerline.append(graphing_array[i][midpoint])
        
    figc, axc = plt.subplots()
    axc.plot(centerline, height)
    py.show()
    print(centerline)
    
    
    
    
    

main()

print("main test - Done.")
