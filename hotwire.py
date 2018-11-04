from helping_classes import Filename, Folder
from data import *
import copy
import pickle

class Hotwire_Data_Reader():
    '''
    Class used to read data from txt files
    '''

    def __init__(self, filename, mode, nb_of_probes=1):
        '''

        :param filename: the name of the file containing the data
        :param mode: if the data was extracted from grid measurements, the data format is slightly different (skiprows)
        :param nb_of_probes: for the case of multiprobe measurements this can be set to 2
        '''
        self.filename = Filename(filename)
        if mode == 'single_hotwire_measurement':
            self.skiprows = 159

        elif mode == 'grid':
            self.skiprows = 0
        else:
            print 'MODE ERROR: %s' % mode

    def read_data(self, active=1):
        '''

        :param active: if active = 1 (default) only one measurement probe is used !!!TO BE IMPROVED!!!
        :return: np array containing the data from the active prope in the form (t, vx, vy)
        '''
        if active == 1:
            data = np.loadtxt(self.filename, skiprows=self.skiprows, unpack=True)
            data = data[:3]
        if active == 2:
            data = np.loadtxt(self.filename, skiprows=self.skiprows, unpack=True)
            data = (data[0], data[3], data[4])
        return data

    def v_wt(self):
        '''

        :return: wind tunnel velocity extracted from filename
        '''
        return self.filename.v()

    def angle(self):
        '''

        :return: angle extracted from filename
        '''
        return self.filename.angle()

    def position(self):
        '''

        :return: posion (x,y,z) extracted from filename
        '''
        return self.filename.position()

    def frequency(self):
        '''

        :return: frequency extracted from filename
        '''
        return self.filename.frequency()


class Hotwire_Data(object):
    '''
    Extract physical quantities from the data files
    '''

    def __init__(self, filename, mode='single_hotwire_measurement', active_probe=1):
        self.data_reader = Hotwire_Data_Reader(filename, mode)
        self._t, self._vx, self._vy = self.data_reader.read_data(active=active_probe)

        self.dt = self.t[1] - self.t[0]
        self.sampling_frequency = 20000.

        self.pos = self.data_reader.position()
        self.angle = self.data_reader.angle()
        self.frequency = self.data_reader.frequency()
        self.v_wt = self.data_reader.v_wt()
        self.caption = {'title': 'Hotwire Measurements',
                        'xlabel': 't / s',
                        'ylabel': '',
                        'legend': '%sdeg %sHz %sms' % (self.angle, self.frequency, self.v_wt)}

    @property
    def t(self):
        return self._t

    @property
    def vx(self):
        caption = self.caption.copy()
        caption['ylabel'] = 'AOA'
        return Timeseries_1D_Periodic_Signal(self._vx, x=self.t, caption=caption, sampling_frequency=1. / self.dt)

    @vx.setter
    def vx(self, value):
        self._vx = value

    @property
    def vy(self):
        caption = self.caption.copy()
        caption['ylabel'] = 'AOA'
        return Timeseries_1D_Periodic_Signal(self._vy, x=self.t, caption=caption, sampling_frequency=1. / self.dt)

    @vy.setter
    def vy(self, value):
        self._vy = value

    def aoa(self, correction=False, **kwargs):
        label = kwargs.pop('label', 'AOA')
        if not correction:
            correction_factor = 0

        aoa = np.arctan2(self._vy - correction_factor, self._vx) / 2 / np.pi * 360
        caption = self.caption.copy()
        caption['ylabel'] = 'AOA'
        AOA = Timeseries_1D_Periodic_Signal(aoa, x=self.t, caption=caption, sampling_frequency=1. / self.dt)
        return AOA

    def v(self, correction=False, **kwargs):
        if correction:
            correction_factor = self.vy.mean()
        else:
            correction_factor = 0
        caption = self.caption.copy()
        caption['ylabel'] = 'v / m/s'
        v = np.sqrt(self._vx ** 2 + (self._vy - correction_factor) ** 2)
        return Timeseries_1D_Periodic_Signal(v, x=self.t, caption=caption, sampling_frequency=1. / self.dt)

    def save(self, filename):
        np.savetxt(fname=filename, X=np.transpose(np.array([self.t, self.vx, self.vy])))


class Hotwire_Data_List(list):
    def __init__(self, *args, **kwargs):
        '''
        Load and Save multiple data files and operate on the data
        :param args:
        :param kwargs:
        '''
        self.delimiter = kwargs.pop('delimiter', 'angle')
        list.__init__(self, *args)

    def load_from_filenames(self, filename_list, mode="single_hotwire_measurement", active_probe=1):
        for f in filename_list:
            self.append(Hotwire_Data(f, mode=mode, active_probe=active_probe))
        return self

    def load_from_folder(self, folder, mode="single_hotwire_measurement", active_probe=1):
        filename_list = Folder(folder).files(absolute=True)
        print 'loading %s files from %s' % (len(filename_list), folder)
        result = self.load_from_filenames(filename_list, mode=mode, active_probe=active_probe)
        return result

    def load_from_pickle(self, filename):
        print 'loading from pickle file %s' % (filename)
        return pickle.load(open(filename, "rb"))

    def pickle(self, filename):
        print 'saving to pickle file %s' % (filename)
        pickle.dump(self, open(filename, "wb"))

    def aoa(self, y_correction=0, **kwargs):
        return Timeseries_1D_Periodic_Signal_List([i.aoa() for i in self])

    def vx(self):
        return Timeseries_1D_Periodic_Signal_List([i.vx for i in self])

    def vy(self):
        return Timeseries_1D_Periodic_Signal_List([i.vy for i in self])

    def v(self, y_correction=0):
        return Timeseries_1D_Periodic_Signal_List([i.v() for i in self])


    # def fit_params(self):
    #     return Hotwire_Data_List([f.fit_params[0][0] for f in self.fits()])
    #
    # def fit_results(self):
    #     return Hotwire_Data_List([f.fit() for f in self.fits()])
    #
    # def psd(self):
    #     return Hotwire_Data_List([i.psd(label=eval('i.' + self.delimiter)) for i in self.aoa()])
    #
    # def autocorrelation(self):
    #     return Hotwire_Data_List([i.autocorrelation(label=eval('i.' + self.delimiter)) for i in self.aoa()])