import time
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt
import statsmodels.api as sm
from scipy.stats import ks_2samp

def compute_p_value_matrix(partitions):
    N = len(partitions)
    # Initialize an NxN matrix to store p-values
    p_value_matrix = np.zeros((N, N))

    # Perform the KS test on each pair of distributions
    for i in range(N):
        for j in range(i, N):
            # KS test between distributions[i] and distributions[j]
            ks_statistic, p_value = ks_2samp(partitions[i], partitions[j])
            p_value_matrix[i, j] = p_value
            p_value_matrix[j, i] = p_value  # Fill the symmetric element

    return p_value_matrix

if __name__ == "__main__":

    start_time = time.time()
    # Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
    # This file has 4096 samples per second
    # starting GPS 1126257415 duration 4096

    y = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    print('Total number of datapoints: ',len(y))
    x = np.linspace(0,4096,4096*4096)
    #plt.figure(0,figsize=(10,4))
    #plt.plot(y,'o',color='violet')
    #plt.title('Entire acquisition')
    #plt.ylabel('Strain')
    #plt.xlabel('Time [s]')

    #plot first second of acquisition
    x1 = np.linspace(0,1,4096)
    y1 = y[:len(x1)]
    plt.figure(1,figsize=(16,6))
    plt.plot(x1,y1,'.',color='slateblue')
    plt.title('First second of acquisition')
    plt.ylabel('Strain')
    plt.xlabel('Time [s]')

    print(y.var())

    # Mean function
    mean_value = np.mean(y)
    print("Mean Function:", mean_value)
    total_std = np.std(y)
    print("Standard Deviation:", total_std)

    #Two-point autocorrelation
    # Compute autocorrelation for multiple lags

    lags = 40000  # Number of lags to include
    autocorr_values, conf = sm.tsa.acf(y, nlags=lags, fft=True,alpha=0.05)
    x_lag = np.arange(0,(lags+1)/4096,1/4096)
    #np.savetxt('data/autocorr_values.txt',autocorr_values)
    #autocorr_values = np.loadtxt('data/autocorr_values.txt')
    plt.figure(2,figsize=(16,6))
    plt.title('Autocorrelation Function')
    plt.plot(x_lag,autocorr_values,'.', color='royalblue')
    #print(autocorr_values)
    #print(1.96*autocorr_values.var())
    #plt.axhline(1.96*np.var(autocorr_values))
    #plt.axhline(1.96/np.sqrt(len(autocorr_values)),color='r')
    confidence_level = 1.96/np.sqrt(len(autocorr_values))
    print(f'confidence level: {conf}')
    print(f'confidence level: {confidence_level}')
    print(conf.shape)
    #plt.fill_between(x_lag,-confidence_level,confidence_level,color='cyan',alpha=0.2)
    plt.fill_between(x_lag,conf[:,0]-autocorr_values,conf[:,1]-autocorr_values,color='red',alpha=0.2)

    plt.xlabel('Lag [s]')


    #Check for stationarity
    mean = []
    std = []
    splits = [0.0001,0.001,0.01,0.1,1,10]
    d = 40936
    splits = [d//10,d//5,d//2,d,2*d,4*d]
    print(splits)
    y_splits = []
    for s in splits:
        timescale = 40936/s
        print(f'timescale: {timescale:.2} s')
        for k in np.array_split(y, s):
            mean.append(np.mean(k))
            std.append(np.std(k))
        plt.figure(f'{s} partitions',figsize=(16,6))
        plt.title(f'Timescale {timescale:.2} s')
        plt.plot(mean/total_std,'o')
        plt.xlabel('Partition #')

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
