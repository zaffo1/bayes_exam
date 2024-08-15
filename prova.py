import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import scipy

def plot_psd_comparison(data):
    fs = 4096
    NFFT = 4*fs  # Use 4 seconds of data for each Fourier transform
    NOVL = int(0.5 * NFFT)  # The number of points of overlap between segments
    psd_window = scipy.signal.windows.tukey(NFFT, alpha=1./4)

    # Compute the PSD using Welch's method
    freqs, Pxx_H1 = welch(data, fs=fs, nperseg=NFFT, window=psd_window, noverlap=NOVL)


    # Define the frequency range of interest (for example, 500 Hz to 1500 Hz)
    freq_range = (150,300)

    # Select the indices corresponding to the desired frequency range
    freq_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]

    # Extract the PSD for the selected frequency range
    Pxx_H1_selected = np.zeros_like(Pxx_H1)
    Pxx_H1_selected[freq_indices] = Pxx_H1[freq_indices]

    # Compute the inverse Fourier transform of the selected PSD to obtain the ACF
    acf = np.fft.irfft(Pxx_H1_selected)

    # Normalize the ACF (optional, depending on what you're looking for)
    acf = acf / acf[0]

    # Plot the PSD
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(freqs, 10 * np.log10(Pxx_H1), label='Original PSD')
    plt.plot(freqs[freq_indices], 10 * np.log10(Pxx_H1[freq_indices]), label='Selected PSD', color='r')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [dB/Hz]')
    plt.legend()
    plt.grid()

    # Plot the ACF
    plt.subplot(1, 2, 2)
    plt.stem(acf)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid()

    plt.tight_layout()
    plt.show()


    exit()
    # Compute the PSD with no window and no averaging
    nowin_freqs, nowin_Pxx_H1,  = welch(data, fs=fs, nperseg=NFFT, window='boxcar')

    plt.figure(figsize=(8, 5))
    ax = plt.axes()
    plt.xscale('log', base=2)
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
    #plt.show()
    # ACF computation (unchanged)
    from statsmodels.tsa.stattools import acf
    acf_array = acf(Pxx_H1, fft=True, nlags=50)
    acf_nowin = acf(nowin_Pxx_H1, fft=True, nlags=50)
    acf_small = acf(Pxx_H1[np.where(freqs == 50)[0][0]:np.where(freqs == 512)[0][0]], fft=True, nlags=50)

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