#Directionality 
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

class Directionality:
    def __init__(self, one):
        self.one = one #placeholder
    
    rotate = matrix_rotation.Rotation(1)
    
    #rotating  mesh of one transducer + middle mesh so that each 
    #transducer will be flat when P is calculated    
    #alpha = array of angles that a transducer is rotated about the z-axis
    #theta = array of angles that a transducer is rotated about the y-axis
    
    #need unconcatenated transducer array
    def full_mesh_rotation(self, t_mesh, m_mesh, alpha, theta):
         #rotate transducer array so that one is flat
         
         #rotate mesh in the same manner as transducer
        
        














