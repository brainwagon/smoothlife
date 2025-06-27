# smoothlife

This is just a basic implementation of the SmoothLifeL automaton, using
thresholds cribbed from the Youtube video at:

http://www.youtube.com/watch?v=KJe9H6qS82I

It uses the fftw library to quickly integrate the weights over the grid,
and then writes out a bunch of pgm files.  You probably can do something 
better with them, but the code will at least serve as a starting point.
