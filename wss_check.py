import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import scipy

def plot_psd_comparison(data):
    fs = 4096
    NFFT = 4 * fs  # Use 4 seconds of data for each Fourier transform
    NOVL = int(0.5 * NFFT)  # The number of points of overlap between segments
    psd_window = scipy.signal.windows.tukey(NFFT, alpha=1./4)

    # Compute the PSD using Welch's method
    freqs, Pxx_H1 = welch(data, fs=fs, nperseg=NFFT, window=psd_window, noverlap=NOVL)

    # Compute the PSD with no window and no averaging
    nowin_freqs, nowin_Pxx_H1,  = welch(data, fs=fs, nperseg=NFFT, window='boxcar')

    plt.figure(figsize=(8, 5))
    ax = plt.axes()
    #plt.xscale('log', base=2)
    plt.yscale('log', base=10)

    # Plotting PSD with and without window
    plt.plot(nowin_freqs, nowin_Pxx_H1, 'red', label='No Window', alpha=.8, linewidth=.7)
    plt.plot(freqs, Pxx_H1, 'blue', label='Tukey Window + Welch Average', alpha=.8, linewidth=.7)

    # Plot 1/f^2 for comparison
    inverse_square = 1 / (nowin_freqs[1:] ** 2)
    scale_index = 500
    scale = nowin_Pxx_H1[scale_index] / inverse_square[scale_index]
    plt.plot(nowin_freqs[1:], inverse_square * scale, 'green', label=r'$1 / f^2$', alpha=.8, linewidth=1)

    plt.axis([20, 1024, 1e-47, 1e-40])
    plt.ylabel(r'S$_n$(t)')
    plt.xlabel('Freq (Hz)')
    ax.tick_params(axis='x', which='minor', direction='out')
    plt.legend(loc='best')
    plt.title('PSD for the full 4096 seconds')
    plt.show()
    exit()
    # ACF computation (unchanged)
    from statsmodels.tsa.stattools import acf
    acf_array = acf(Pxx_H1, fft=True, nlags=50)
    acf_nowin = acf(nowin_Pxx_H1, fft=True, nlags=50)
    acf_small = acf(Pxx_H1[np.where(freqs == 50)[0][0]:np.where(freqs == 300)[0][0]], fft=True, nlags=50)

    plt.plot(acf_array,'o', label='Windowed')
    plt.plot(acf_nowin,'o', label='Leakage')
    plt.plot(acf_small,'o', label='Windowed+Restricted')
    plt.ylabel('Normalized ACF')
    plt.xlabel('Lag')
    plt.legend(loc='best')
    plt.title('Normalized ACF for the PSD')
    plt.show()

    return



if __name__ == '__main__':
    t0 = 1126257415
    t1 = 1126259462.4
    hdata = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    plot_psd_comparison(hdata)