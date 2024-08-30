# Code adapted from https://github.com/gwastro/PyCBC-Tutorials/blob/master/tutorial/3_WaveformMatchedFilter.ipynb
import numpy
import matplotlib.pyplot as plt
from pycbc import types, fft
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from pycbc.types.timeseries import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.filter import matched_filter, highpass, sigma


if __name__ == '__main__':
	t0 = 1126257415
	t1 = 1126259462.4
	hdata = numpy.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
	strain = TimeSeries(hdata, delta_t=1/4096, epoch=t0) # epoch = Time of the first sample in seconds.

	# 1. Preconditioning the data
	# supress low freqeuncy behavior which can introduce numerical artefacts
	strain = highpass(strain, 15.0)

	#plt.plot(strain.sample_times, strain)
	#plt.xlabel('Time (s)')
	#plt.show()

    # Remove given seconds from either end of time series (to remove spike in the data at the boundaries)
	conditioned = strain.crop(2, 2)

	#plt.plot(conditioned.sample_times, conditioned)
	#plt.xlabel('Time (s)')
	#plt.show()

	# 2. Estimate the power spectral density

	# Optimal matched filtering requires weighting the frequency components of the potential
	# signal and data by the noise amplitude.

	# We use 4 second samples of our time series in Welch method.
	psd = conditioned.psd(4)

	# Now that we have the psd we need to interpolate it to match our data
	# and then limit the filter length of 1 / PSD. After this, we can
	# directly use this PSD to filter the data in a controlled manner
	psd = interpolate(psd, conditioned.delta_f)

	# 1/PSD will now act as a filter with an effective length of 4 seconds
	# Since the data has been highpassed above 15 Hz, and will have low values
	# below this we need to inform the function to not include frequencies
	# below this frequency.
	psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate),
									low_frequency_cutoff=15)


	# 3. make your signal model

	# Conceptually, matched filtering involves laying the potential signal over your
	# data and integrating (after weighting frequencies correctly).
	# If there is a signal in the data that aligns with your 'template',
	# you will get a large value when integrated over.

	# In this case we "know" what the signal parameters are. In a search
	# we would grid over the parameters and calculate the SNR time series
	# for each one
	Masses = []
	peaks = []
	times = []

	M_chirp = 25.29#23.28
	q = 0.83#0.84

	'''
	# Get a frequency domain waveform
	sptilde, sctilde = get_fd_waveform(approximant="TaylorF2",
							 mass1=mass1_from_mchirp_q(M_chirp, q),
							 mass2=mass2_from_mchirp_q(M_chirp, q),
							 delta_f=1.0/4,
							 f_lower=20)

	# FFT it to the time-domain
	delta_t = 1/4096
	tlen = int(1.0 / delta_t / sptilde.delta_f)
	sptilde.resize(tlen/2 + 1)
	sp = TimeSeries(types.zeros(tlen), delta_t=delta_t)
	fft.ifft(sptilde, sp)
	'''

	sp, sc = get_td_waveform(approximant="TaylorF2",
						mass1=mass1_from_mchirp_q(M_chirp, q),
						mass2=mass2_from_mchirp_q(M_chirp, q),
						delta_t=conditioned.delta_t,
						f_lower=20)


	# We will resize the vector to match our data
	sp.resize(len(conditioned))

	# The waveform begins at the start of the vector, so if we want the
	# SNR time series to correspond to the approximate merger location
	# we need to shift the data so that the merger is approximately at the
	# first bin of the data.

	# This function rotates the vector by a fixed amount of time.
	# It treats the data as if it were on a ring. Note that
	# time stamps are *not* in general affected, but the true
	# position in the vector is.

	# By convention waveforms returned from `get_td_waveform` have their
	# merger stamped with time zero, so we can use the start time to
	# shift the merger into position
	template = sp.cyclic_time_shift(sp.start_time)


	#4. calculating the signal-to-noise time series

	snr = matched_filter(template, conditioned,
					 psd=psd, low_frequency_cutoff=20)

	# Remove time corrupted by the template filter and the psd filter
	# We remove 4 seonds at the beginning and end for the PSD filtering
	# And we remove 4 additional seconds at the beginning to account for
	# the template length (this is somewhat generous for
	# so short a template). A longer signal such as from a BNS, would
	# require much more padding at the beginning of the vector.
	snr = snr.crop(4 + 4, 4)

	# Why are we taking an abs() here?
	# The `matched_filter` function actually returns a 'complex' SNR.
	# What that means is that the real portion correponds to the SNR
	# associated with directly filtering the template with the data.
	# The imaginary portion corresponds to filtering with a template that
	# is 90 degrees out of phase. Since the phase of a signal may be
	# anything, we choose to maximize over the phase of the signal.
	plt.figure(figsize=[12, 4])
	plt.plot(snr.sample_times, abs(snr), color = 'navy')
	plt.title('The SNR computed for the full timeseries')
	plt.ylabel('SNR')
	plt.xlabel('Time (s)')
	plt.tight_layout()
	plt.savefig('figures/snr.png')
	plt.show()

	peak = abs(snr).numpy().argmax()
	snrp = snr[peak]
	time = snr.sample_times[peak]
	Masses.append((M_chirp, q))
	peaks.append(snrp)
	times.append(time)

	print(f"We found a signal at {time}s with SNR {abs(snrp)}")

	# 5. Aligning and Subtracting the Proposed Signal
	# In the previous section we found a peak in the signal-to-noise for a proposed binary black hole merger.
	# We can use this SNR peak to align our proposal to the

	# Shift the template to the peak time
	dt = time - conditioned.start_time
	aligned = template.cyclic_time_shift(dt)

	# scale the template so that it would have SNR 1 in this data
	aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)

	# Scale the template amplitude and phase to the peak value
	aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
	aligned.start_time = conditioned.start_time


	# 6. Visualize the overlap between the signal and data
	#To compare the data an signal on equal footing,
	# and to concentrate on the frequency range that is important.
	# We will whiten both the template and the data,
	# and then bandpass both the data and template between 30-300 Hz.
	# In this way, any signal that is in the data is transformed in the same way that the template is.

	# We do it this way so that we can whiten both the template and the data
	white_data = (conditioned.to_frequencyseries() / psd**0.5).to_timeseries()
	white_template = (aligned.to_frequencyseries() / psd**0.5).to_timeseries()

	white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
	white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

	# Select the time around the merger
	white_data = white_data.time_slice(t1-.2, t1+.1)
	white_template = white_template.time_slice(t1-.2, t1+.1)

	plt.figure(figsize=[12, 5])
	plt.plot(white_data.sample_times, white_data, label="Data")
	plt.plot(white_template.sample_times, white_template, label="TaylorF2", color='crimson')
	plt.xlabel("Time (s)")
	plt.title("Comparison between data and model (whitened and bandpassed)")
	plt.legend()
	plt.tight_layout()
	plt.savefig('figures/data_vs_model.png')
	plt.show()
