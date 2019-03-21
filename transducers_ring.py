#Acoustic levitation
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
    
#actually three rings of transducers at 3 different radia.
#radius should be 5mm 
class Transducers:
    def __init__(self, ntr, radius, meshN):
        self.radius = radius
        self.meshN = meshN
        self.ntr = ntr  #transducers in rx
        
    def transducer_angle(self, tr):
        angle = 2 * np.pi / (tr)
        return angle
    
    
    # creates circular mesh
    def ring_points(self):
        centerx = 0      
        centery = 0       
                            
        x_start = centerx - self.radius
        x_end   = centerx + self.radius
           
        y_start = centery - self.radius
        y_end   = centery + self.radius
        #mesh point created as a square
        x = np.linspace(x_start, x_end, self.meshN) 
        y = np.linspace(y_start, y_end, self.meshN)
        
        x, y = np.meshgrid(x, y)
        
        #defining points inside the radius, and only keeping those
        r = np.sqrt((x - centerx)**2 + (y - centery)**2)
        inside = r < self.radius 
        
        return x[inside], y[inside]
    
    # returns mesh (x,y,z) coordinates of all of the transducers in one ring
    # coordinate][number of transducer]
    # 0 = x, 1 = y, 2=z
    def transducer(self):  # 3D
        #create a square mesh and only pick values that are in circle
        xy = self.ring_points()
        x = xy[0] ; y = xy[1]
                
        # array for z values
        # all zeros initially before translation
        height = np.zeros(len(x))
        
        #holds x and y values of transducer meshes
        transducer_i_x = []
        transducer_i_y = []
        transducer_i_z = []
        
        #creating meshpoints for y axis transducers
        for i in range(0, self.ntr): 
            # creation of ntr number of transducers at the origin
            transducer_i_x.append(x)
            transducer_i_y.append(y)
            transducer_i_z.append(height)
        
            #for graphing one transducer
        """   
        fig, ax = plt.subplots()
        ax.set(xlabel='X', ylabel='Y', aspect=1.0)
        
        ax.scatter(transducer_i_x[0], transducer_i_y[0]) 
        plt.show()  
        
        print("called")
        """
        
        return transducer_i_x, transducer_i_y, transducer_i_z
    

#Transducers(1, 5, 11).ring_points()

#for testing    
#how to access data :
#called three times, once for each ring
'''
#how to grab data
ntr = [6, 12, 18]
for i in range(len(ntr)): 
    all_t = Transducers(ntr, 5, 10)
    print(all_t.z) #broken
'''
    
'''
fig, ax = plt.subplots()
ax.set(xlabel='X', ylabel='Y', aspect=1.0)   
N = [6,12,18]
for n in N:
    
    all_t = Transducers(n, 5, 10)
    all_t = all_t.transducer()
    #[0 = x, 1 = y, 2=z][transducer_i]
    x = all_t[0][0]  
    y = all_t[1][0]  
    
    for i in range(n):  
        ax.scatter(all_t[0][i], all_t[1][i]) 
plt.show()  
'''
'''
fig, ax = plt.subplots()
ax.set(xlabel='X', ylabel='Y', aspect=1.0)
#how to grab just one ring:
all_t = Transducers(6, 5, 1, 50, 0)
all_t = all_t.transducer()
x = all_t[0][0]
y = all_t[1][0]
ax.scatter(x, y)
plt.show()
'''







print("transducers_ring - Done.")