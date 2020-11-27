'''
This code determines the pressure field between two spherical caps of transducers. 
Each spherical cap is comproised of a given number of transducers (using ntr), 
and the height of each ring of transducers must also be specified. 

Other parameters such as the number of mesh points that comprise the transducers
or the measurement space can also be configured. 

This code was my introduction to scientific computing and working with multiple files. 
It was also my attempt at learning OOP. Because of this, the file structure and the use
of classes is a bit wacky. I also had not yet learned the standards of practice for 
commenting within the code. These are changes I could make, but the code is functional 
and it is nice to see how much I have learned by looking back at this.
'''
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

import matrix_rotation
import transducers_ring
import tiny_lev_positions2 as tlp
import measurement_mesh
import matrix_method
import rotated_mesh


#initial condition constants
ntr = [6,12,18,6,12,18]                                
ntr_l = [6,12,18]
ntr_u = [6,12,18]
h_i = [0, 0.8, 1.6, 119.1, 118.3, 117.5]
h_l = [0, 0.8, 1.6] ; h_u = [119.1, 118.3, 117.5]
r_i = [10.52, 21.35, 30.62, 10.52, 21.35, 30.62]
r_l = [10.52, 21.35, 30.62]; r_u = r_l

# radius of transducer [mm]
t_radius = 4.5                
# number of points in side length of square that represents transducer
t_meshN = 8
# number of points in side length of square that represents measurement space
m_meshN = 60  

z_middle = 59.55                  
radius_largest_ring = 30.62    #

# Constants
# frequency of emitted sound [Hz]
omega = 2*np.pi*40.e3    
# wave propogation velocity [mm/s]           
c = 343.e3                  
amplitude = 1     
amplitude2 = 1
phase = np.pi                
phase2 = 0
# density of propogation medium [kg/mm^3]
dens = 1.225e-9        
# wavelength of emitted sounds [mm]     
wavelength = (c/omega) 

t_mesh = len(transducers_ring.Transducers(1, t_radius, t_meshN).ring_points()[0])
ntr_half = np.array([ntr[0],ntr[1],ntr[2]])

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
        return np.concatenate(x), np.concatenate(y), np.concatenate(z)
    
    def measurement_mesh_full():
        x = [] ; y = [] ; z = []
        m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                         int(m_meshN), 0, t_radius).m_mesh()
        for i in range(int(len(m_mesh[0][0]) / 2)):
            x.append(m_mesh[0][i]) 
            y.append(m_mesh[1][i])
            z.append(m_mesh[2][i])
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
                
        return np.concatenate(x), np.concatenate(y), np.concatenate(z)
    
    #actually gives a 2D slice of the 3D mesh
    def half_mesh_m(shift=0):
        x = [] ; y = [] ; z = []
    
        m_mesh = measurement_mesh.M_mesh(radius_largest_ring, h_i[2], z_middle,
                                         int(m_meshN), 2, t_radius, shift=shift).m_mesh()
    
        for i in range(len(m_mesh[0][0])):
            
            x.append(m_mesh[0][i]) 
            y.append(m_mesh[1][i])
            z.append(m_mesh[2][i])
        
        return np.concatenate(x), np.concatenate(y), np.concatenate(z)
    
    
    #graphing transducer mesh and middle mesh
    def directional_m_mesh(shift = 0):
        '''
        shift = amount we are shifting the rotated meshes away from the origin
        '''
        xm_ar = [] ; ym_ar = [] ; zm_ar = [] #these store the values of arrays for calculating P | SUPER IMPORTANT
        
        for i in range(len(ntr)):
                    
            if i <= int(len(ntr) / 2 - 1):
                rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                                radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).rotated_middle_func(0, shift=shift)
            else:
                rotated_middle = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                                                radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).rotated_middle_func(1, shift=shift)
           
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
                            radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).unrotated_rings(0)
            else:
                xyz_t = rotated_mesh.Rotated_mesh(ntr[i], z_middle, h_i[i], t_meshN,
                            radius_largest_ring, m_meshN, t_radius, h_i[2], h_i[3], r_i[i]).unrotated_rings(1)        
            for j in range(ntr[i]):
                
                x = xyz_t[j][0] #; x = np.concatenate((x , half_mesh_m()[0]))
                y = xyz_t[j][1] #; y = np.concatenate((y , half_mesh_m()[1]))
                z = xyz_t[j][2] #; z = np.concatenate((z , half_mesh_m()[2]))
                
                x_ar.append(x) ; y_ar.append(y) ; z_ar.append(z)
                
        return x_ar, y_ar, z_ar

    '''
    ACCESSING DATA: -- pre concatenate --
    translated_tm_mesh()[0,1,2 = transister x/y/z | 3,4,5 = middle xyz][transducer i]
    '''
    # now calculating pressure matrix using moved interior matrices and transducers
    # at the origin
    shift = np.linspace(0, 20, 35)
    scrolling_graph = []
    for origin_i in range(len(shift)):
        xyz_t = directional_t_mesh()
        xyz_m = directional_m_mesh(shift=shift[origin_i])
        '''
        NOTE: because the matix multiplication is the same for transducers across the 
        origin, we are just oging to calculate for postive transducers then flip the 
        pressure matrix and add this to the pressure matrix.
        
        '''
        
        m_meth = matrix_method.Matrix_method(omega, c, amplitude, t_mesh, t_radius,
                                            phase, dens, wavelength)
        m_meth2 = matrix_method.Matrix_method(omega, c, amplitude2, t_mesh, t_radius,
                                            phase2, dens, wavelength)
        

        # calculate the transfer and excitation matrices
        p = np.zeros((len(xyz_m[0][0]),1), dtype = complex)
        

        # Functional pressure matrix with p_flip!
        # NOTE: according to the matrix method paper, "When the acoustic wave
        # reaches the transducer, it is reflected, and the constant
        # ωρc/λ should be replaced by j/λ
        for k in range(int(len(ntr) / 1)):
            if k == 0:
                i = 0
            else:
                i = 0
                for l in range(k):
                    i += ntr[k - l - 1]
            if k < 3:        
                for j in range(int(ntr[k] / 2)):
                    #transducer to measurement point
                    t_points = ([xyz_t[0][i]] + [xyz_t[1][i]] + [xyz_t[2][i]])  
                    m_points = ([xyz_m[0][i]] + [xyz_m[1][i]] + [xyz_m[2][i]]) 
                    
                    transfer_matrix = m_meth.t_matrix(t_points, m_points)
                    u_matrix = m_meth.u_matrix(t_points)
                    p_matrix = m_meth.p_matrix(transfer_matrix, u_matrix)
                    
                    p[:][:] += p_matrix[:][:]

                    # Optional additions
                    '''
                    #transducer to transducer to measurement point
                    #matrix method always multiplies by (wpc/leambda)
                    #calculating T^tm (m x n_top) matrix
                    #r stands for reflector and transducer, since we only have
                    #transducers as reflectors
                    #m_points_c = half_mesh_m()
                    tmpts = half_mesh_t(1)
                    t_meshp = ([tmpts[0]] + [tmpts[1]] + [tmpts[2]])
                    transfer_matrix_rm = m_meth.t_matrix(t_meshp, m_points)
                    transfer_matrix_tt = m_meth.t_matrix(t_points, t_meshp)
                    t_matrix_ttm = np.matmul(transfer_matrix_rm, 
                                            transfer_matrix_tt)
                    p_matrix = 1.j / wavelength * m_meth.p_matrix(t_matrix_ttm,
                                                                u_matrix) 
                    
                    p[:][:] += p_matrix[:][:]
                    '''
                    '''
                    #TM RT TR U additon
                    transfer_matrix_rt = m_meth.t_matrix(t_meshp, t_points)
                    t_matrix_tmrt = np.matmul(transfer_matrix, transfer_matrix_rt)
                    t_matrix_tmrttr = np.matmul(t_matrix_tmrt, transfer_matrix_tt)
                    p_matrix = (1.j / wavelength)**2 * m_meth.p_matrix(t_matrix_tmrttr, 
                            u_matrix)
                    
                    p[:][:] += p_matrix[:][:]
                    '''
                    '''
                    #third addition from matrix method paper
                    t_m_rmtr = np.matmul(transfer_matrix_rm, transfer_matrix_tt)
                    t_m_rmtrrt = np.matmul(t_m_rmtr, transfer_matrix_rt)
                    t_m_rmtrrttr = np.matmul(t_m_rmtrrt, transfer_matrix_tt)
                    p_matrix = (1.j/ wavelength)**3 * m_meth.p_matrix(t_m_rmtrrttr,
                            u_matrix)
                    p[:][:] += p_matrix[:][:]
                    
                    #fourth addition
                    t_m_tmrt = np.matmul(transfer_matrix, transfer_matrix_rt)
                    t_m_tmrttr = np.matmul(t_m_tmrt, transfer_matrix_tt)
                    t_m_tmrttrrt = np.matmul(t_m_tmrttr, transfer_matrix_rt)
                    t_m_tmrttrrttr = np.matmul(t_m_tmrttrrt, transfer_matrix_tt)
                    p_matrix = (1.j/ wavelength)**4 * m_meth.p_matrix(t_m_tmrttrrttr, 
                            u_matrix)
                
                    p[:][:] += p_matrix[:][:]
                    '''
        
                    i += 1
                    
                    
            if k > 2:
                for j in range(int(ntr[k] / 2)):
                    t_points = ([xyz_t[0][i]] + [xyz_t[1][i]] + [xyz_t[2][i]])  
                    m_points = ([xyz_m[0][i]] + [xyz_m[1][i]] + [xyz_m[2][i]]) 
                    
                    transfer_matrix = m_meth2.t_matrix(t_points, m_points)
                    u_matrix = m_meth2.u_matrix(t_points)
                    p_matrix = m_meth2.p_matrix(transfer_matrix, u_matrix)
                                    
                    p[:][:] += p_matrix[:][:] 

                    # Optional additions
                    '''
                    tmpts = half_mesh_t(0)
                    t_meshp = ([tmpts[0]] + [tmpts[1]] + [tmpts[2]])
                    transfer_matrix_tm = m_meth2.t_matrix(t_meshp, m_points)
                    transfer_matrix_tt = m_meth2.t_matrix(t_points, t_meshp)
                    t_matrix_ttm = np.matmul(transfer_matrix_tm, 
                                            transfer_matrix_tt)
                    p_matrix = 1.j / wavelength * m_meth2.p_matrix(t_matrix_ttm,
                                                                u_matrix) 
                    
                    p[:][:] += p_matrix[:][:]
                    '''
                    '''
                    #TM RT TR U additon
                    transfer_matrix_ttr = m_meth2.t_matrix(t_meshp, t_points)
                    t_matrix_tmrt = np.matmul(transfer_matrix, transfer_matrix_ttr)
                    t_matrix_tmrttr = np.matmul(t_matrix_tmrt, transfer_matrix_tt)
                    p_matrix = (1.j / wavelength)**2 * m_meth2.p_matrix(t_matrix_tmrttr, 
                            u_matrix)
                    
                    p[:][:] += p_matrix[:][:]
                    '''
                    '''
                    #third addition from matrix method paper
                    t_m_rmtr = np.matmul(transfer_matrix_rm, transfer_matrix_tt)
                    t_m_rmtrrt = np.matmul(t_m_rmtr, transfer_matrix_rt)
                    t_m_rmtrrttr = np.matmul(t_m_rmtrrt, transfer_matrix_tt)
                    p_matrix = (1.j/ wavelength)**3 * m_meth2.p_matrix(t_m_rmtrrttr,
                            u_matrix)
                    p[:][:] += p_matrix[:][:]
                    
                    #fourth addition
                    t_m_tmrt = np.matmul(transfer_matrix, transfer_matrix_rt)
                    t_m_tmrttr = np.matmul(t_m_tmrt, transfer_matrix_tt)
                    t_m_tmrttrrt = np.matmul(t_m_tmrttr, transfer_matrix_rt)
                    t_m_tmrttrrttr = np.matmul(t_m_tmrttrrt, transfer_matrix_tt)
                    p_matrix = (1.j/ wavelength)**4 * m_meth2.p_matrix(t_m_tmrttrrttr, 
                            u_matrix)
                
                    p[:][:] += p_matrix[:][:]
                    '''
                    i += 1
        # obtaining the modulus of the pressure
        p = np.absolute(p) 
        
        xyz_m = half_mesh_m()
        
        # Plot the pressure map
        # Need Pressure array to now be 2D with x = x and y = z
        
        #creating the pressure array used for graphing
        xy_size = int(np.sqrt(len(xyz_m[0])))
        graphing_array = np.reshape(p, (xy_size, xy_size))
        print(np.shape(graphing_array), 'shape of graphing array')
        p_flipped = np.empty_like(graphing_array) # discussed in LaTeX document
        for i in range(len(graphing_array[0])):
            for j in range(int(len(graphing_array[0]) / 2)):
                p_flipped[i][j] = graphing_array[i][int(len(graphing_array) - 1 - j)]
                p_flipped[i][int(len(graphing_array) - 1 - j)] = graphing_array[i][j]
        
        graphing_array = [graphing_array + p_flipped for graphing_array,
                        p_flipped in zip(graphing_array, p_flipped)]   
        
        # these will be the x and z values of our M space...must be a mesh though
        # so the space before concatenated!
        # NOTE: ONLY WORKS FOR HALF MESH
        X = np.reshape(xyz_m[0], (xy_size, xy_size)) #only works for half mesh
        Z = np.reshape(xyz_m[2], (xy_size, xy_size))
        
        print('working on plotting')

    

        fig = pyplot.figure(figsize=(11,7), dpi=100)
        plt.pcolormesh(X, Z, graphing_array, alpha=0.5, cmap=cm.viridis)
        pyplot.colorbar()
        #pyplot.contour(X, Z, graphing_array, cmap=cm.viridis)
        #pyplot.streamplot(X, Z, u, v)
        pyplot.xlabel('X mm')
        pyplot.ylabel('Z mm')
        pyplot.title("Pressure Field: y = {} mm".format(round(shift[origin_i], 2)))
        py.show()
        # Important code for generating specific graphs
        # ---------------------------------------------
        # #fitting levitation points to centerline
        # xp  = tlp.Pos(0).lev_pos()[0]
        # yp  = tlp.Pos(0).lev_pos()[1]
        
        # #finding the center points used for graphing
        # centerline = []
        # midpoint = int(m_meshN / 2)
        # height = np.linspace(h_l[-1], h_u[-1], m_meshN)
        # for i in range(len(graphing_array[0])):
        #     centerline.append(graphing_array[i][midpoint])
        
        # for i in range(len(height)):
        #     for j in range(len(yp)):
        #         if height[i] - yp[j] < 0.2 and height[i] - yp[j] > 0:
        #             if xp[j] == 0:
        #                 xp[j] = centerline[i]
            
        # figc, axc = plt.subplots()
        # axc.plot(centerline, height)
        # axc.scatter(xp, yp)
        #py.show()
        # ---------------------------------------------
        #py.savefig(str(origin_i))

main()



print("main - Done.")
