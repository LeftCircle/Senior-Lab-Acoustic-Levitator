# Senior-Lab-Acoustic-Levitator
It works!

The following code can be used to calculate the pressure resulting from an array of transducers which are arranged as two spherical caps, where each transducer is pointing towards the center point between the transducer arrays. 

## Acoustic Levitation Papers

The code was created using the matrix method for determining the acoustic radiation force, which can be found [here](https://www.researchgate.net/publication/224254694_Matrix_Method_for_Acoustic_Levitation_Simulation)

## Functionality

Currently the code only supports a spherical cap composed of three rings of transducers at three different heights. It also only supports an even number of transducers in each ring. This is due to complications in the matrix method from the transducers being positioned so that they emmit sound at an angle. 
