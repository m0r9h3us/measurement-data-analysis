# Measurement Data Analysis

Some classes and functions used for the analysis of hotwire data.

## hotwire.py
Contains a class to load the data from x-wire measurements in the wind tunnel. It enables the calculation of basic quantities as Angle of Attack (AOA), Velocity (v), from the measured components vx and vy. All Data returned by this class is defined in data.py 

## data.py
Data classes are derived from np.ndarray. By adding additional arguments to ndarray, direct plotting in the way data.plot() shall become easier but remain flexible. 
