#Acoustic levitation
import numpy as np
import pylab as py
import matplotlib.pyplot as plt

'''
1.) Calculate the pressure field with Rayleigh integral
discretize transducer and reflector into small cells
'''
#create however many transducers from center to radius and place on y axis
    #translate each one vertically in z to match position
    #rotate alpha degrees about x axis to match angle
    #copy each transducer, rotate in polar to real position, then rotate about
    #x to get actual orientation
    
#actually three rings of transducers at 3 different radia.
#radius should be 5mm and distance between ~2.5 mm
class Transducers:
    #just creating the transducers on z for now
    def __init__(self, tr1, radius,
                 z, meshN, yshift):
        #self.centerx = centerx
        #self.centery = centery       
        self.radius = radius
        self.meshN = meshN
        #self.distance_b_transducers = distance_b_transducers
        self.tr1 = tr1  #transducers in rx
        self.z = z
        self.yshift = yshift
    #Transducers in inner ring (ring1)
    def transducer_angle(self, tr):
        angle = 2 * np.pi / (tr)
        return angle
    
    # returns mesh (x,y,z) coordinates of all of the transducers in one ring
    # coordinate][number of transducer]
    # 0 = x, 1 = y, 2=z
    def transducer(self):  # 3D
        #create a square mesh and only pick values that are in circle
        #holds x and y values of transducer meshes
        transducer_i_x = []
        transducer_i_y = []
        transducer_i_z = []
                
        #creating meshpoints for y axis transducers
        for i in range(0, self.tr1): 
            #radius_ring1 = (2. * self.radius / (self.transducer_angle(self.tr1)))*1.5
            centerx = 0#radius_ring1 * np.cos(self.transducer_angle(self.tr1)*i)
            centery = 0#radius_ring1 * np.sin(self.transducer_angle(self.tr1)*i) - (self.yshift) #quick fix for positioning
                                
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
            
            height = np.zeros(len(x[inside]))
            #height[:] += self.z
            transducer_i_x.append(x[inside])
            transducer_i_y.append(y[inside])
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
    
    

#for testing    
#how to access data :
#called three times, once for each ring
'''
#how to grab data
all_t = Transducers(6, 5, 1, 50, 0)
print(all_t.z)
'''
    
'''
fig, ax = plt.subplots()
ax.set(xlabel='X', ylabel='Y', aspect=1.0)   
N = [6,12,18]
for n in N:
    
    all_t = Transducers(n, 5, 0, 50, 0)
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