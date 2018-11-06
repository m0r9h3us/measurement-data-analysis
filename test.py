from data import *
from helping_classes import *
from hotwire import *

path = "/home/morpheus/Projects/measurement_data_analysis/Test_Data/"
# Test class Hotwire_Data():
caption = {'xlabel': 't / s',
           'ylabel': "Caption 1",
           'legend': 'bla'}

caption2 = {'xlabel': 't / s',
            'ylabel': 'Caption 2',
            'legend': 'bla2'}

# # -----------------------------------------------------------------------------------------------------------------
# # Test Object creation and basic plotting
# b = Data_1D(np.arange(10), x=np.arange(10), caption=caption)
# c = Timeseries_1D(np.arange(10), x=np.arange(10), caption=caption, sampling_frequency=20000).plot()
# d= Timeseries_1D_Periodic_Signal(np.arange(10), x=np.arange(10), caption=caption2, sampling_frequency=20000).plot()
# e= Timeseries_1D_Periodic_Signal(0.5*np.arange(10), x=np.arange(10), caption=caption2, sampling_frequency=20000).add_to_plot(d, args=['ro-'])

# -----------------------------------------------------------------------------------------------------------------
# Test hotwire.py and basic plotting
hot = Hotwire_Data(filename=path + "4Hz_5deg_30ms_pos1_posx0_posy0_posz0.txt")

AOA = hot.aoa()  # ok
AOA.plot() # ok
V = hot.v().plot() # ok
Vx = hot.vx.plot() # ok
Vy = hot.vy.plot() # ok
MOV = AOA.moving_average(window_size=1000)

# # -----------------------------------------------------------------------------------------------------------------
# # Test data.py and data manipulation methods
# p = AOA.plot()
# # moving average
# MOV = AOA.moving_average(window_size=1000) #ok
# MOV.add_to_plot(p) # ok
#
# PHA = AOA.phase_average(HOT.sampling_frequency / HOT.frequency) # ok
# PHA.plot() # ok
#
# FFT = AOA.fft() #ok
# FFT.plot() #ok

# Test Fitting
# p = AOA.moving_average(window_size=1000).plot() # ok
# AOA.sinus_fit(p0=[hot.angle, hot.frequency,0])[1].add_to_plot(p) # ok

# --------------------------------------------------------------------------------------------------------------------
### Test Hotwire_Data_List

# # Load from folder and save as pickle
#Data = Hotwire_Data_List().load_from_folder(path)
#Data.pickle(path+"Test_Data.pickle")

# Load from pickle
# Data = Hotwire_Data_List().load_from_pickle(path+"Test_Data.pickle")
# Data_AOA = Data.aoa()
# Data_AOA_MOV= Data_AOA.moving_average(window_size=1000)
# multiplot = Data_AOA_MOV.plot()