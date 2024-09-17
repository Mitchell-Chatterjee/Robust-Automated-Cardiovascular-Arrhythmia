from scipy import signal
from abc import ABC, abstractmethod


class Filter(ABC):
    def __init__(self, filt):
        self.filter = filt

    @abstractmethod
    def apply_filter(self, sig):
        """
        Apply the filter to the given signal
        :return: The signal after applying the filter.
        """


class ButterworthFilter(Filter):

    def __init__(self, low, high, fs):
        """
        Creates a new butterworth filter.
        :param low: The low frequency cutoff.
        :param high: The high frequency cutoff.
        """
        super().__init__(signal.butter(N=3, Wn=[low, high], btype='bandpass', fs=fs, output='ba'))

    def apply_filter(self, sig):
        """
        Applies the butterworth filter to the given signal
        :param sig: The signal to apply the filter to.
        :return: The signal after applying butterworth filtering.
        """
        b, a = self.filter
        return signal.lfilter(b, a, sig)


class MedianFilter(Filter):
    def __init__(self):
        """
        Creates a new median filter.
        """
        super().__init__(signal.medfilt)

    def apply_filter(self, sig):
        """
        Apply median filter to the given signal.
        :param sig: The signal to filter.
        :return: The signal after applying median filtering.
        """
        return self.filter(sig)
