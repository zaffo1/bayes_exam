import numpy as np
import matplotlib.pyplot as plt


# Two-point autocorrelation function
def autocorrelation(y, lag):
    n = len(y)
    mean_value = np.mean(y)
    autocorr = np.sum((y[:n-lag] - mean_value) * (y[lag:] - mean_value))
    autocorr /= np.sum((y - mean_value) ** 2)
    return autocorr

if __name__ == "__main__":

    # Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
    # This file has 4096 samples per second
    # starting GPS 1126257415 duration 4096
    y = np.loadtxt('H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    print(len(y))
    plt.figure(1)
    plt.plot(y)
    #plt.show()

    # Mean function
    mean_value = np.mean(y)
    print("Mean Function:", mean_value)

    #Compute autocorrelation for lag=1
    lag = 1
    autocorr_value = autocorrelation(y, lag)
    print(f"Two-Point Autocorrelation for lag={lag}:", autocorr_value)

    # Optionally, compute autocorrelation for multiple lags
    lags = np.arange(0,10000,100)
    print(lags)
    autocorr_values = [autocorrelation(y, lag) for lag in lags]
    print("Autocorrelation values for different lags:", autocorr_values)


    plt.figure(2)
    plt.plot(lags,autocorr_values)
    plt.show()
