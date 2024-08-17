from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import kstest
from scipy.signal import welch
import scipy

if __name__=='__main__':
    t0 = 1126257415
    t1 = 1126259462 #time of the event
    # The event occurred at GPS time 1126259462 = September 14 2015, 09:50:45 UTC
    # Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
    # This file has 4096 samples per second
    # starting GPS 1126257415 duration 4096
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')

    #PSD using welch's method
    fs = 4096
    NFFT = 4 * fs  # Use 4 seconds of data for each Fourier transform
    NOVL = int(0.5 * NFFT)  # The number of points of overlap between segments
    psd_window = scipy.signal.windows.tukey(NFFT, alpha=1./4)

    plt.figure(figsize=(8, 5))
    ax = plt.axes()
    plt.xlim(0,fs/2)
    #plt.ylim(1e-48,1e-40)
    #plt.xscale('log', base=2)
    plt.title('Power Spectral Density for the full 4096 seconds')
    plt.yscale('log', base=10)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'S$_n$(f)')
    # Compute the PSD using Welch's method
    freqs, Pxx_H1 = welch(data, fs=fs, nperseg=NFFT, window=psd_window, noverlap=NOVL)

    # Plotting PSD with and without window
    plt.plot(freqs, Pxx_H1, alpha=.8, linewidth=1)
    plt.savefig('figures/pds_full.png')
    plt.show()


    plt.figure(figsize=(8, 5))
    ax = plt.axes()
    plt.xlim(16,512)
    plt.ylim(1e-47,1e-41)
    plt.xscale('log', base=2)
    plt.title('Power Spectral Density for the full 4096 seconds')
    plt.yscale('log', base=10)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'S$_n$(f)')
    # Compute the PSD using Welch's method
    freqs, Pxx_H1 = welch(data, fs=fs, nperseg=NFFT, window=psd_window, noverlap=NOVL)

    # Plotting PSD with and without window
    plt.plot(freqs, Pxx_H1, alpha=.8, linewidth=1)
    plt.savefig('figures/pds_full_logscale.png')
    plt.show()


    plt.figure(figsize=(12, 7))
    ax = plt.axes()
    plt.xlim(16,1024)
    plt.ylim(1e-47,1e-40)
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    n_split = 10
    for j, k in enumerate(np.array_split(data, n_split)):
        # Compute the PSD using Welch's method
        freqs, Pxx_H1 = welch(k, fs=fs, nperseg=NFFT, window=psd_window, noverlap=NOVL)
        # Plotting PSD with and without window
        plt.plot(freqs, Pxx_H1, label=f'Partition bin #{j+1}', alpha=.8, linewidth=.7)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'S$_n$(f)')
    plt.legend()
    plt.savefig('figures/pds_partition.png')
    plt.show()