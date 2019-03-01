#matrix rotation
import numpy as np
import pylab as py
import matplotlib.pyplot as plt

class Rotation:
    def __init__(self, one):
        #self.z_middle = z_middle #midpoint of TinyLev
        self.one = one #placeholder
    
    #need to rotate each of the transducers in ring 1 then 2 then 3 
    #assuming theta is in radians and not degrees ( I mean who uses degrees right) .... right?
    #def angle_from_zmiddle(self, radius_transducer_i):
    #    angle_from_z = np.arctan(radius_transducer_i / self.z_middle)
    #    return angle_from_z
    
    #xyz = [transducer[0][0], transducer[1][0], transducer[2][0]] #how to define xyz of transducer_i
    def rotation_x(self, array, theta):
        rx = np.ndarray(shape = (3,3), dtype = float)
        rx[0] = [1, 0, 0]
        rx[1] = [0, np.cos(theta), -np.sin(theta)]
        rx[2] = [0, np.sin(theta),  np.cos(theta)]
        
        xrot_array = np.matmul(rx, array)
        
        return xrot_array
    
    def rotation_y(self, array, theta):
        ry = np.ndarray(shape = (3,3), dtype = float)
        ry[0] = [np.cos(theta), 0, np.sin(theta)]
        ry[1] = [0, 1, 0]
        ry[2] = [-np.sin(theta), 0,  np.cos(theta)]
        
        yrot_array = np.matmul(ry, array)
        return yrot_array
    
    def rotation_z(self, array, theta):
        rz = np.ndarray(shape = (3,3), dtype = float)
        rz[0] = [np.cos(theta), -np.sin(theta), 0]
        rz[1] = [np.sin(theta), np.cos(theta), 0 ]
        rz[2] = [0, 0,  1]
        
        zrot_array = np.matmul(rz, array)
        return zrot_array
    
    def rotation_u(self, xc, yc, zc, array, theta): 
        #input same y and negative x for first quadrant
        ru = np.ndarray(shape = (3,3), dtype = float)
        ru[0] = [np.cos(theta) + xc**2 * (1 - np.cos(theta)),
                 xc * yc * (1 - np.cos(theta)) - zc * np.sin(theta),
                 xc * zc * (1 - np.cos(theta)) + yc * np.sin(theta)]
        ru[1] = [yc * xc * (1 - np.cos(theta)) + zc * np.sin(theta),
                 np.cos(theta) + yc**2 * (1 - np.cos(theta)),
                 yc * zc * (1 - np.cos(theta)) - xc * np.sin(theta)]
        ru[2] = [zc * xc * (1 - np.cos(theta)) - yc * np.sin(theta),
                 zc * yc * (1 - np.cos(theta)) + xc * np.sin(theta),
                 np.cos(theta) + zc**2 * (1 - np.cos(theta))]
        return np.matmul(ru, array)
        
        #turned out to be useless
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        