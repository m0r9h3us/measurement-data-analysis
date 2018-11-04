import numpy as np
from helping_classes import Plot
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


class Data_1D(np.ndarray):
    '''
    Adding some attributes to the np.array class
    '''

    def __new__(cls, input_array, x=None, x_error=None, y_error=None, caption=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if not x is None and not input_array.size == x.size:
            raise ValueError('Dimensions do not fit')
        if not x_error is None and not input_array.size == x_error.size:
            raise ValueError('Dimensions do not fit')
        if not y_error is None and not input_array.size == y_error.size:
            raise ValueError('Dimensions do not fit')
        obj._x = x
        obj._x_error = x_error
        obj._y_error = y_error
        obj.caption = caption
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._x = getattr(obj, '_x', None)
        self._x_error = getattr(obj, '_x_error', None)
        self._y_error = getattr(obj, '_y_error', None)
        caption_default = {'title': 'Data_1D',
                           'xlabel': '',
                           'ylabel': '',
                           'legend': ''}
        self.caption = getattr(obj, 'caption', caption_default)
        # no need to return anything

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if not value.size == self.size:
            raise ValueError(
                'Array size does not fit! Size of Data is %s and size of x is %s' % (value.size, self.size))
        else:
            self._x = value

    #
    # @property
    # def x_error(self):
    #     return self._x_error
    #
    # @x_error.setter
    # def x_error(self, value):
    #     if not value.size == self.size:
    #         raise ValueError(
    #             'Array size does not fit! Size of Data is %s and size of xerrir is %s' % (value.size, self.size))
    #     else:
    #         self._x_error = value
    #
    # @property
    # def y_error(self):
    #     return self._y_error
    #
    # @y_error.setter
    # def x(self, value):
    #     if not value.size == self.size:
    #         raise ValueError(
    #             'Array size does not fit! Size of Data is %s and size of y_error is %s' % (value.size, self.size))
    #     else:
    #         self._y_error = value

    def moving_average(self, window_size):
        window = np.ones(int(window_size)) / int(window_size)
        result = np.convolve(self, window, 'same')
        caption = self.caption.copy()
        caption['legend'] = caption['legend'] + ' MA'
        return Data_1D(result, x=self.x, caption=caption)

    def autocorrelation(self):
        autocorr = signal.fftconvolve(self, self[::-1], mode='full')
        normalized_autocorrelation = autocorr[autocorr.size / 2:] / autocorr[autocorr.size / 2]
        steps = np.linspace(-len(autocorr), len(autocorr), 1)
        return Data_1D(normalized_autocorrelation, x=np.arange(), caption='ACF')
        # return Autocorrelation(self).calc(**kwargs)

    def increment(self, shift):
        result = np.roll(self, - shift)[:-shift] - self[:-shift]
        return Data_1D(result, caption=str(self.caption) + 'inc', x=np.arange(len(result)))

    def detrend(self, global_mean=True):
        if global_mean:
            return Timeseries_1D(self - self.mean())

    def split(self, phase_length):
        '''
        Replace with np.array_split!
        :param phase_length:
        :return:
        '''
        phase_length = int(phase_length)
        nb_of_chunks = self.size / phase_length
        if not np.mod(self.size, phase_length) == 0:
            cut = self.size - np.mod(self.size, phase_length)
            data = self[:cut]
        elif np.mod(self.size, phase_length) == 0:
            data = self[:]
        split_array = np.array(np.split(data, nb_of_chunks))
        return split_array

    def plot(self, args=['b-'], plotpath=None, filename=None, subplots=None, **kwargs):
        '''
        Make plotting possible, without the need but the possibility to configure everything manually
        :param args: list - List of arguments passed directly to the plot function
        :param plotpath: string - for saving
        :param filename: string - for saving
        :param subplots: Hm not the best IDEA
        :param kwargs: dict - Overwrite all settings (title, xlabel, ylabel, legend)
        :return: tuple - (fig, ax) to manipulate the plot afterwards
        '''
        # PLOT AOA( Angle) MIT RMSE for each frequency
        if not subplots:
            fig, ax = plt.subplots()
        else:
            fig, ax = subplots

        caption = self.caption.copy()
        title = self.caption.get('title', 'Data_1D.plot()')
        xlabel = self.caption.get('xlabel', 'x')
        ylabel = self.caption.get('ylabel', 'y')
        legend = self.caption.get('legend', '')

        # kwargs overwrite settings
        title = kwargs.pop('title', title)
        xlabel = kwargs.pop('xlabel', xlabel)
        ylabel = kwargs.pop('ylabel', ylabel)
        legend = kwargs.pop('legend', legend)

        plot_function = ax.plot  # alternativly errorbar
        if not isinstance(self.x, np.ndarray):
            plot_function(self, *args, label=legend)
        else:
            plot_function(self.x, self, *args, label=legend)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend(loc='best', fancybox=True)

        # for file_format in fileformats:
        #    fig.savefig(plotpath + self.caption['legend'], dpi=100)
        plt.show()

        return (fig, ax)

    def add_to_plot(self, subplots, args=['r-'], **kwargs):
        '''
        Takes an existing and adds new content
        :param subplots: tuple (fig,ax)
        :param args: list - List of arguments passed directly to the plot function
        :param kwargs: Allow overwrite here? Otherwise remove!
        :return:
        '''
        legend = self.caption.get('legend', '')
        legend = kwargs.pop('legend', legend)

        fig, ax = subplots

        if not isinstance(self.x, np.ndarray):
            ax.plot(self, *args, label=legend)
        else:
            ax.plot(self.x, self, *args, label=legend)
        ax.legend()
        plt.show()


class Timeseries_1D(Data_1D):
    def __new__(cls, input_array, x=None, x_error=None, y_error=None, caption=None, sampling_frequency=20000.):
        if not isinstance(x, np.ndarray):
            print 'array x not initialized'
            x = np.linspace(0, input_array.size / np.float(sampling_frequency), input_array.size)
        obj = Data_1D.__new__(cls, input_array, x=x, x_error=x_error, y_error=y_error, caption=caption)
        obj.sampling_frequency = sampling_frequency
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._x = getattr(obj, '_x', None)
        self._x_error = getattr(obj, '_x_error', None)
        self._y_error = getattr(obj, '_y_error', None)
        caption_default = {'title': 'Timeseries_1D',
                           'xlabel': 't / s',
                           'ylabel': '',
                           'legend': ''}
        self.caption = getattr(obj, 'caption', caption_default)
        self.sampling_frequency = getattr(obj, 'sampling_frequency', None)

    def t(self):
        '''

        :return: ndarray - Time axes based on array size and saved sampling frequency
        '''
        return np.arange(self.size) / np.float(self.sampling_frequency)

    def fft(self, zero_padding=True, normalize=False):
        '''
        Calculate the Fourier Transform of the Time-Series
        returns positive spectrum part ONCE
        '''

        signal = self - self.mean()
        N = signal.size
        if not zero_padding:
            freq = np.fft.fftfreq(N, d=1 / self.sampling_frequency)
            fft = np.fft.fft(signal)
        elif zero_padding:
            freq = np.fft.fftfreq(2 ** (int(np.log2(N)) + 3), d=1 / self.sampling_frequency)[0:N]
            fft = np.fft.fft(signal, 2 ** (int(np.log2(N)) + 3))[0:N]

        if normalize:
            fft = 2. / N * np.abs(fft)

        caption = self.caption.copy()
        caption['legend'] = caption['legend'] + ' FFT'
        caption['xlabel'] = 'f / Hz'
        caption['ylabel'] = 'FFT'
        return Data_1D(fft, x=freq, caption=caption)

    def psd(self, **kwargs):
        fft = self.fft(**kwargs)
        psd = 2 * np.abs(fft) ** 2
        caption = self.caption.copy()
        caption['legend'] = caption['legend'] + ' PSD'
        caption['xlabel'] = 'f / Hz'
        caption['ylabel'] = 'PSD'
        return Data_1D(psd, x=fft.x, caption=caption, sampling_frequency=self.sampling_frequency)

    def TI(self, averaging_length=4000):
        nb_of_chunks = int(self.size / averaging_length)
        rest = np.mod(self.size, averaging_length)
        a = np.array(np.array_split(self[:-rest], nb_of_chunks)[:-1])
        TI = self.std() / self.mean()
        return TI


class Timeseries_1D_Periodic_Signal(Timeseries_1D):
    '''
    Used for the analysis of periodic wind tunnel excitations
    '''

    def __new__(cls, input_array, x=None, x_error=None, y_error=None, caption=None, sampling_frequency=20000.):
        '''
        For inheritance from np.ndarray.
        :param input_array: np.ndarray
        :param x: np.ndarray - Adds x as an attribute to the ndarray
        :param x_error: np.ndarray - Adds x_error as an attribute to the ndarray
        :param y_error: np.ndarray - Adds y_error as an attribute to the ndarray
        :param caption: dict - Definition of some Quantities necessary for Plotting
        :param sampling_frequency: float - Might be replace by another dictionay containing the measurement infos (angle etc.)
        :return: obj is used in array finalize
        '''
        if not isinstance(x, np.ndarray):
            print 'array x not initialized'
            x = np.linspace(0, input_array.size / np.float(sampling_frequency), input_array.size)
        obj = Timeseries_1D.__new__(cls, input_array, x=x, x_error=x_error, y_error=y_error, caption=caption)
        obj.sampling_frequency = sampling_frequency
        return obj

    def __array_finalize__(self, obj):
        '''
        For inheritance from np.ndarray.
        :param obj: The result from __new__ call
        :return: None
        '''
        if obj is None: return
        self._x = getattr(obj, '_x', None)
        # print self._x
        self._x_error = getattr(obj, '_x_error', None)
        self._y_error = getattr(obj, '_y_error', None)
        caption_default = {'title': 'Timeseries_1D_Periodic_Signal',
                           'xlabel': 't / s',
                           'ylabel': '',
                           'legend': ''}
        self.caption = getattr(obj, 'caption', None)
        self.sampling_frequency = getattr(obj, 'sampling_frequency', None)

    def sinus_fit(self, p0):
        '''
        Fit the measurement data with a simple sinus-function. Uses scipy.curve_fit
        :param p0: list - [Amplitude, Frequency, Phase- Shift]
        :return: tuple - first parameter are the fitting parameter returned by scipy, second is an np.ndarray
        '''
        fit_funct = lambda t, A, f, phi0: A * np.sin(2 * np.pi * f * t - phi0)
        fit_params = curve_fit(fit_funct, self.t(), self, p0=p0)
        # print 'A: %s, f: %s, phi: %s' % (fit_params[0][0], fit_params[0][1], fit_params[0][2])
        caption = self.caption.copy()
        caption['legend'] = caption['legend'] + ' FIT'
        fit = Timeseries_1D_Periodic_Signal(fit_funct(self.t(), fit_params[0][0], fit_params[0][1], fit_params[0][2]),
                                            x=self.t(), caption=caption)
        return (fit_params, fit)

    def TI(self, averaging_length=4000):
        nb_of_chunks = int(self.size / averaging_length)
        rest = np.mod(self.size, averaging_length)
        a = np.array(np.array_split(self[:-rest], nb_of_chunks)[:-1])
        TI = a.std(axis=1) / a.mean(axis=1)
        return TI

    # for periodic signals
    def phase_average(self, phase_length, **kwargs):
        '''
        Calculate the phase average
        :param phase_length:
        :param kwargs:
        :return:
        '''
        split_arr = self.split(phase_length=phase_length)
        phase_average = split_arr.mean(axis=0)
        phase_std = split_arr.std(axis=0)
        x = np.arange(len(phase_average)) * self.sampling_frequency
        caption = self.caption.copy()
        caption['legend'] = caption['legend'] + ' PA'
        return Timeseries_1D(phase_average, x=x, y_error=phase_std, caption=caption,
                             sampling_frequency=self.sampling_frequency)

    def validate(self, phase_length, plot=True):
        '''
        Used to check for measurement errors in the data
        :param phase_length:
        :param plot:
        :return:
        '''
        start = 3
        split_arr = self.split(phase_length=phase_length)

        error = [0 for i in np.arange(start)]
        if plot:
            fig, ax = plt.subplots()
        for i, j in enumerate(split_arr[3:]):
            diff = j - split_arr[3]
            error.append(np.sum(np.abs(diff)))
            if plot:
                ax.plot(j)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(Data(error))
        return Timeseries_1D(error)


class Data_1D_List(list):
    def __init__(self, *args):
        list.__init__(self, *args)
        self.x = [i.x for i in self]

    def moving_average(self, window_size):
        return [a.moving_average(window_size=window_size, caption=str(a.caption) + 'mov. avg.') for a in self]

    def phase_average(self, phase_length):
        return [a.phase_average(caption=a.caption, phase_length=phase_length) for a in self]

    def autocorrelation(self):
        return [a.autocorrelation(caption=a.caption) for a in self]

    def psd(self):
        return [a.psd(caption=a.caption) for a in self]

    def mean(self):
        return [i.mean() for i in self]

    def increment(self, shift):
        return [i.increment(shift) for i in self]

# class FFT():
#     '''
#     Gibt nur EINMAL den positiven Anteil des Spektrums zurck
#     '''
#
#     def __init__(self, data):
#         self.data = data
#         self.signal = data - data.mean()
#
#     def calc(self, zero_padding=True, normalize=False, **kwargs):
#         caption = kwargs.pop('caption', self.data.caption)
#         N = len(self.signal)
#         if not zero_padding:
#             freq = np.fft.fftfreq(N, d=self.data.sampling_frequency)
#             fft = np.fft.fft(self.signal)
#         elif zero_padding:
#             freq = np.fft.fftfreq(2 ** (int(np.log2(N)) + 3), d=self.data.sampling_frequency)[0:N]
#             fft = np.fft.fft(self.signal, 2 ** (int(np.log2(N)) + 3))[0:N]
#
#         if normalize:
#             fft = 2. / N * np.abs(fft)
#         return Data(fft, x=freq, caption=caption, sampling_frequency=self.data.sampling_frequency)
#


# class PSD():
#     def __init__(self, data):
#         self.data = data
#         self.fft_obj = FFT(self.data)  # for debugging
#         self.fft = self.fft_obj.calc(zero_padding=True)
#
#     def calc(self, **kwargs):
#         caption = kwargs.pop('caption', self.data.caption)
#         psd = 2 * np.abs(self.fft) ** 2
#         return Data(psd, x=self.fft.x, caption=caption, sampling_frequency=self.data.sampling_frequency)


# class Autocorrelation():
#     def __init__(self, data):
#         self.data = data
#         self.signal = data - data.mean()
#
#     def calc(self, **kwargs):
#         caption = kwargs.pop('caption', self.data.caption)
#         N = len(self.data)
#         d = N * np.ones(2 * N - 1)
#         # acf = (np.correlate(signal, signal, 'full') / d)
#         acf = (np.correlate(self.signal, self.signal, 'full') / d)
#         acf = acf[N - 1:]
#         acf_normalized = acf / self.data.std() ** 2  # np.std(signal) ** 2
#         step = np.arange(N)
#         return Data(acf_normalized, x=step, caption=caption, sampling_frequency=self.data.sampling_frequency)

# class Fit():
#     def __init__(self, data, moving_average=False, p0=[5, 5, 0]):
#         '''Data: data type object'''
#         self.data = data
#         if moving_average:
#             self.data = self.data.moving_average(window_size=1000)
#         self.fit_funct = lambda t, A, f, phi0: A * np.sin(2 * np.pi * f * t - phi0)
#         self.fit_params = curve_fit(self.fit_funct, self.data.x, self.data,
#                                     p0=p0)
#
#     def fit(self, **kwargs):
#         '''
#
#         :return: Data Object
#         '''
#         aoa_fit = self.fit_funct(self.data.x, self.fit_params[0][0], self.fit_params[0][1],
#                                  self.fit_params[0][2])
#         return Data(aoa_fit, x=self.data.x,
#                     caption='Fit (%.2f,%.2f,%.2f)' % (
#                         self.fit_params[0][0], self.fit_params[0][1], self.fit_params[0][2]),
#                     sampling_frequency=self.data.sampling_frequency)
#         # self.phase_length = int(1. / self.aoa_fit_params[0][1] * self.measurement_frequency)


# class Phase_Average():
#     def __init__(self, data, phase_length):
#         self.data = data
#         self.phase_length = int(phase_length)
#         self.nb_of_chunks = len(self.data.t()) / self.phase_length
#         print 'nb of chunks: %s' % self.nb_of_chunks
#         if not np.mod(len(self.data), self.phase_length) == 0:
#             cut = len(self.data) - np.mod(len(self.data), self.phase_length)
#
#             self.data = self.data[:cut]
#         self.nb_of_chunks = len(self.data.t()) / self.phase_length
#         print 'nb of chunks: %s' % self.nb_of_chunks
#         print 'mod %s' % (np.mod(len(self.data), self.phase_length))
#
#     def calc(self, **kwargs):
#         caption = kwargs.pop('caption', self.data.caption)
#         # print caption
#         split_arr = np.array(np.split(self.data, self.nb_of_chunks))
#         phase_average = split_arr.mean(axis=0)
#         phase_std = split_arr.std(axis=0)
#
#         # phase_average = np.zeros(self.phase_length)
#         # for i in np.arange(len(split_arr)):
#         #     phase_average = np.add(phase_average, split_arr[i])
#         #     #if i<100:
#         #     #plt.plot(phase_average)
#         # phase_average = phase_average / self.nb_of_chunks
#         # print 'laenge: %s, dx %s'%(len(phase_average),self.data.dx[0])
#         # print self.data.x[0]
#         return Data(phase_average, x=np.arange(len(phase_average)) * self.data.sampling_frequency,
#                     y_error=phase_std, caption=caption, sampling_frequency=self.data.sampling_frequency)
