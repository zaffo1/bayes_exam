import pylab
from pycbc.catalog import Merger
from pycbc import types, fft
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from pycbc.types.timeseries import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.filter import matched_filter
from pycbc.filter import sigma
import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == '__main__':
	t0 = 1126257415
	t1 = 1126259462.4
	hdata = numpy.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')

	strain = TimeSeries(hdata, delta_t=1/4096, epoch=t0) # epoch = Time of the first sample in seconds.

	strain = highpass(strain, 15.0)
	# strain = resample_to_delta_t(strain, 1.0/2048)

	pylab.plot(strain.sample_times, strain)
	pylab.xlabel('Time (s)')
	pylab.show()

    # Remove given seconds from either end of time series
	conditioned = strain.crop(2, 2)

	pylab.plot(conditioned.sample_times, conditioned)
	pylab.xlabel('Time (s)')
	pylab.show()

	# Estimate the power spectral density

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




    #Modify a PSD such that the impulse response associated with its inverse square root is no
    # longer than max_filter_len time samples.
    # In practice this corresponds to a coarse graining or smoothing of the PSD.
	psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate),
									low_frequency_cutoff=15)


	M_chirp = numpy.linspace(25, stop=35, num=100)
	q = numpy.linspace(0.5, stop=1, num=10)
	Masses = []
	peaks = []
	times = []
	m = 36 # Solar masses
	# for i in tqdm(range(0, len(M_chirp))):
		# for j in range(0, len(q)):
			# Get a frequency domain waveform
	M_chirp = 25.34
	q = 0.83

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

			# Resize the vector to match our data
	sp.resize(len(conditioned))
	template = sp.cyclic_time_shift(sp.start_time)
			# print("starting")

	hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
					 mass1=36,
					 mass2=36,
					 delta_t=conditioned.delta_t,
					 f_lower=20)
	hp.resize(len(conditioned))
	template1 = hp.cyclic_time_shift(hp.start_time)

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

	# pylab.figure(figsize=[10, 4])
	pylab.plot(snr.sample_times, abs(snr))
	pylab.title('The SNR computed for the full timeseries')
	pylab.ylabel('Signal-to-noise')
	pylab.xlabel('Time (s)')
	# pylab.show()


	peak = abs(snr).numpy().argmax()
	snrp = snr[peak]
	time = snr.sample_times[peak]
	Masses.append((M_chirp, q))
	peaks.append(snrp)
	times.append(time)

	print("We found a signal at {}s with SNR {}".format(time,
														abs(snrp)))
	'''
	Masses = numpy.asarray(Masses)
	peaks = numpy.asarray(peaks)
	times = numpy.asarray(times)
	numpy.save("Masses", Masses)
	numpy.save("peaks", peaks)
	numpy.save("times", times)

	n, bins, patches = plt.hist(times, 50, density=True, facecolor='g', alpha=0.75)
	plt.xlabel('GPS Times')
	plt.ylabel('#')
	plt.title('Histogram of SNR peaks')
	plt.show()
	# The time, amplitude, and phase of the SNR peak tell us how to align
	# our proposed signal with the data.
	'''

	# Shift the template to the peak time
	dt = time - conditioned.start_time
	aligned = template.cyclic_time_shift(dt)
	aligned1 = template1.cyclic_time_shift(dt)

	# scale the template so that it would have SNR 1 in this data
	aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)
	aligned1 /= sigma(aligned1, psd=psd, low_frequency_cutoff=20.0)

	# Scale the template amplitude and phase to the peak value
	aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
	aligned.start_time = conditioned.start_time
	aligned1 = (aligned1.to_frequencyseries() * snrp).to_timeseries()
	aligned1.start_time = conditioned.start_time

	# We do it this way so that we can whiten both the template and the data
	white_data = (conditioned.to_frequencyseries() / psd**0.5).to_timeseries()
	white_template = (aligned.to_frequencyseries() / psd**0.5).to_timeseries()
	white_template1 = (aligned1.to_frequencyseries() / psd**0.5).to_timeseries()

	white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
	white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)
	white_template1 = white_template1.highpass_fir(30, 512).lowpass_fir(300, 512)

	# Select the time around the merger
	white_data = white_data.time_slice(t1-.2, t1+.1)
	white_template = white_template.time_slice(t1-.2, t1+.1)
	white_template1 = white_template1.time_slice(t1-.2, t1+.1)

	pylab.figure(figsize=[15, 3])
	pylab.plot(white_data.sample_times, white_data, label="Data")
	pylab.plot(white_template1.sample_times, white_template1, label="SEOBNRv4_opt")
	pylab.plot(white_template.sample_times, white_template, label="TaylorF2")
	pylab.xlabel("Time (s)")
	pylab.title("Comparison between data and models (whitened and bandpassed)")
	pylab.legend()
	pylab.show()
	'''
	subtracted = conditioned - aligned

	# Plot the original data and the subtracted signal data

	for data, title in [(conditioned, 'Original H1 Data'),
						(subtracted, 'Signal Subtracted from H1 Data')]:

		t, f, p = data.whiten(4, 4).qtransform(.001, logfsteps=100, qrange=(8, 8), frange=(20, 512))
		pylab.figure(figsize=[15, 3])
		pylab.title(title)
		pylab.pcolormesh(t, f, p**0.5, vmin=1, vmax=6)
		pylab.yscale('log')
		pylab.xlabel('Time (s)')
		pylab.ylabel('Frequency (Hz)')
		pylab.xlim(t1 - 2, t1 + 1)
		pylab.show()

	'''