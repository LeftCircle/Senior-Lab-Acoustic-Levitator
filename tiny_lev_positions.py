#Physical Levitation Positions
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#heights in notches of the tinylev with x wire on top
#red = np.array((17, 19.5, 15, 8, 9, 9.5, 13, 16, 11, 12, 10, 14, 16.5, 16))
#blue = np.array((9,6, 5.5, 4, 9.5, 7, 8, 11, 17))

red = np.array((17, 19, 15, 8, 9, 13, 16, 11, 12, 10, 14, 16))
blue = np.array((9,6, 5, 4, 9, 7, 8, 11, 17))


#conversion from notches to mm  +/- 0.635mm
def notch_conv(lev):
    for i in range(len(lev)):
        lev[i] = 4.8*lev[i]
    return lev

#variables for heights in mm
redmm = notch_conv(red)
bluemm = notch_conv(blue)

#flipping the blue to be oriented with the red wire on top based off of 
#highest measurement mesh value
def blue_to_red(lev):
    for i in range(len(lev)):
        lev[i] = 119.1 - lev[i]
    return lev
#blue flipped and now in mm
blue_rmm = blue_to_red(bluemm)

#combining the red and blue
lev_spots = np.concatenate((redmm, blue_rmm))

#defining the error - this is just a random yerr
yerr = 2

#zeros for graphing
zero_r = np.zeros(len(red)+len(blue))

#plotting with error
plt.figure()
plt.errorbar(zero_r, lev_spots, yerr=yerr, fmt='o')
#plt.errorbar(3*4.8, 22*4.8, xerr = yerr, yerr=yerr, fmt = 'o')
plt.show()
