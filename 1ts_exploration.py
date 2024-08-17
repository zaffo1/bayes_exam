from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import kstest

if __name__=='__main__':
    t0 = 1126257415
    t1 = 1126259462 #time of the event
    # The event occurred at GPS time 1126259462 = September 14 2015, 09:50:45 UTC
    # Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
    # This file has 4096 samples per second
    # starting GPS 1126257415 duration 4096
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    print(len(data))
    print(len(data)/4096,'total seconds')
    print(4096**2)
    print(t1-t0)
    strain = TimeSeries(data, t0=t0, sample_rate=4096, unit='strain')

    #print(strain)
    print(kstest(data, 'norm'))

    #plot = strain.plot()
    #ax = plot.gca()
    #ax.set_ylabel('Gravitational-wave amplitude [strain]')
    #ax.set_epoch(1126259462)
    #ax.set_title('LIGO-Hanford strain data around GW150914')
    #ax.axvline(t1, color='orange', linestyle='--')
    #plot.refresh()
    #plot.savefig('figures/whole_timeseries.png')
    #plot.show()

    #sample mean
    sample_mean = strain.mean()
    print(f'Sample mean = {sample_mean}')

    #two-point autocorrelation
    plt.figure('acf',figsize=(16,6))
    autocorr_values = sm.tsa.acf(data, nlags=4096, fft=True)
    #autocorr_values, conf = sm.tsa.acf(data, nlags=1000, fft=True,alpha=0.05)
    #plt.fill_between(np.arange(len(autocorr_values)),conf[:,0]-autocorr_values,conf[:,1]-autocorr_values,color='red',alpha=0.2)

    plt.title('Autocorrelation Function')
    plt.plot(autocorr_values,'.', color='royalblue')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.savefig('figures/acf.png')
    plt.show()

    exit()
    #Amplitude Spectral Density
    lasd2 = strain.asd(fftlength=8, method="median")
    plot = lasd2.plot()
    ax = plot.gca()
    ax.set_ylabel('Amplitude Spectral Density')
    ax.set_xlim(10, 1400)
    ax.set_ylim(1e-24, 1e-20)
    plot.show(warn=False)

    #Power Spectral Density
    psd2 = strain.psd(fftlength=8, method="median")
    plot = psd2.plot()
    ax = plot.gca()
    ax.set_ylabel('Power Spectral Density')
    ax.set_xlim(10, 1400)
    #ax.set_ylim(1e-24, 1e-20)
    plot.show(warn=False)