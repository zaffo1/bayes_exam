import time
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt
import statsmodels.api as sm



def next_pow_two(n):
    """
    find the next power of 2 given n
    """
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorrelation(x, norm=True):
    """
    compute the autocorrelation function of an array
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


if __name__ == "__main__":

    start_time = time.time()
    # Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
    # This file has 4096 samples per second
    # starting GPS 1126257415 duration 4096

    y = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    print('Total number of datapoints: ',len(y))
    y = y[:100000]
    plt.figure(1)
    plt.plot(y)

    print(y.var())

    # Mean function
    mean_value = np.mean(y)
    print("Mean Function:", mean_value)


    # Compute autocorrelation for multiple lags


    # Compute autocorrelation
    lags = 2000  # Number of lags to include
    autocorr_values = sm.tsa.acf(y, nlags=lags, fft=True)
    #np.savetxt('data/autocorr_values.txt',autocorr_values)
    #autocorr_values = np.loadtxt('data/autocorr_values.txt')
    plt.figure(2)
    plt.plot(autocorr_values,'o', color='red')
    plt.xlabel('Lag')



    print(autocorr_values)
    plt.show()
    exit()
    # Augmented Dickey-Fuller test
    adf_result = smt.adfuller(y)

    # Results interpretation
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical Values:", adf_result[4])

    if adf_result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")


    print("--- %s seconds ---" % (time.time() - start_time))

    plt.show()
