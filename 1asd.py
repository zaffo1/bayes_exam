from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from scipy.stats import kurtosis, skew, kstest, moment
from scipy.signal import correlate, get_window
from scipy.signal.windows import tukey

if __name__=='__main__':
    t0 = 1126257415
    t1 = 1126259462 #time of the event
    # The event occurred at GPS time 1126259462 = September 14 2015, 09:50:45 UTC
    # Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
    # This file has 4096 samples per second
    # starting GPS 1126257415 duration 4096
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    print(len(data))
    print(len(data)/4996,'total seconds')
    print(4996**2)
    print(t1-t0)
    strain = TimeSeries(data, t0=t0, sample_rate=4096, unit='strain')

    print(strain)

    plot = strain.plot()
    ax = plot.gca()
    ax.set_ylabel('Gravitational-wave amplitude [strain]')
    ax.set_epoch(1126259462)
    ax.set_title('LIGO-Hanford strain data around GW150914')
    ax.axvline(t1, color='orange', linestyle='--')
    plot.refresh()
    plot.show()


    #Amplitude Spectral Density
    lasd2 = strain.asd(fftlength=8, method="median")
    plot = lasd2.plot()
    ax = plot.gca()
    ax.set_ylabel('Amplitude Spectral Density')
    ax.set_xlim(10, 1400)
    ax.set_ylim(1e-24, 1e-20)
    plot.show(warn=False)