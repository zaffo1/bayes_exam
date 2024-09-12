import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
from scipy.stats import chi2
import scipy
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window
from statsmodels.tsa.stattools import acf

t0 = 1126257415
time_of_event = 1126259462.4
data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
ldata = TimeSeries(data, t0=t0, sample_rate=4096, unit='strain')
print(ldata)

specgram = ldata.spectrogram(16,fftlength=4, overlap=2, window='tukey') ** (1/2.)
plot = specgram.plot()


ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(10, 1400)
ax.colorbar(
    clim=(1e-24, 1e-20),
    norm="log",
    label=r"Strain noise [$1/\sqrt{\mathrm{Hz}}$]",
)
plot
plot.show()




exit()

ldata = ldata.crop(time_of_event-2,time_of_event+2)
gps = time_of_event


hq = ldata.q_transform(frange=(30, 500))
plot = hq.plot()
ax = plot.gca()
ax.set_epoch(gps)
ax.set_yscale('log')
ax.colorbar(label="Normalised energy")

plot.show()

exit()

# ACF computation (unchanged)
acf_array = acf(fftamp, fft=True, nlags=50)
#acf_small = acf(Pxx_H1[np.where(freqs == 40)[0][0]:np.where(freqs == 300)[0][0]], fft=True, nlags=50)

plt.figure(figsize=(10, 4))
plt.plot(acf_array,'o', label='Full PSD', color='slateblue')
#plt.plot(acf_small,'o', label='PSD considering interval 40-300 Hz', color='blueviolet')
plt.ylabel('ACF')
plt.xlabel('Lag')
plt.legend(loc='best')
plt.title('ACF computed for the PSD')
plt.tight_layout()

plt.show()