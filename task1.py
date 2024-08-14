from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from scipy.stats import kurtosis, skew, kstest, moment
from scipy.signal import correlate, get_window
from scipy.signal.windows import tukey

def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


if __name__=='__main__':
    t0 = 1126257415
    t1 = 1126259462.4
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    c = int(len(data)/8000)
    print(c)
    x = np.linspace(0, 0.5, num=c)
    plt.plot(x, data[c:c*2])
    plt.plot(x, data[80000:c+80000])
    plt.ylabel("[strain]")
    plt.xlabel("Time [s]")
    plt.title("Comparison of short time intervals")
    plt.show()

    kstest(data[c:c*2], 'norm')

    mean = []
    std = []
    skew_l = []
    kurtosis_l = []
    moment_3 = []
    for k in np.array_split(data, 16):
        mean.append(np.mean(k))
        std.append(np.std(k))
        skew_l.append(skew(k))
        kurtosis_l.append(kurtosis(k))
        moment_3.append(moment(k, 4))
        # for j in np.array_split(data, 5):
        #     kstest(k, j)
    plt.plot(mean,color='red')
    plt.plot(moment_3)
    plt.show()

    data1 = data

    print(np.mean(data[:c]))
    print(np.std(data[:c]))
    print(skew(data[:c]))
    print(np.mean(data[c:]))
    print(np.std(data[c:]))
    print(skew(data[c:]))
    acf_array = acf(data, fft=True, nlags=10000)
    plt.plot(acf_array)
    plt.show()

    #corr = correlate(data, data1, method='fft')
    #arr = plt.acorr(data, maxlags=100)
    #plt.plot(arr)
    #plt.show()
    strain = TimeSeries(data, t0=t0, sample_rate=4096, unit='strain')

    lasd2 = strain.asd(fftlength=4, method="median")
    plot = lasd2.plot()
    ax = plot.gca()
    ax.set_xlim(10, 1400)
    ax.set_ylim(1e-24, 1e-20)
    plot.show(warn=False)

    psd = strain.psd(fftlength=4*4096, overlap=2*4096, window=tukey(4*4096, alpha=1./4))
    plot = psd.plot()
    ax = plot.gca()
    ax.set_yscale('log')
    ax.set_ylim(10, 1400)
    ax.colorbar(
    clim=(1e-24, 1e-20),
    norm="log",
    label=r"Strain noise [$1/\sqrt{\mathrm{Hz}}$]",
    )
    center = int(t1)
    strain1 = strain.crop(center-16, center+(16))
    fig2 = strain1.asd(fftlength=8).plot()
    plt.show()
    fig1 = strain.plot()
    # plt.show()
    white_data = strain.whiten()
    bp_data = white_data.bandpass(30, 400)
    fig3 = bp_data.plot()
    plt.show()