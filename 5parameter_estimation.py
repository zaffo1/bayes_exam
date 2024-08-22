#https://github.com/gw-odw/odw-2023/blob/main/Tutorials/Day_3/Tuto_3.2_Parameter_estimation_for_compact_object_mergers.ipynb

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

import bilby
from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio
from bilby.core.prior import Uniform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters
from gwpy.timeseries import TimeSeries


if __name__=='__main__':
    t0 = 1126257415
    time_of_event = 1126259462.4
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    strain = TimeSeries(data, t0=t0, sample_rate=4096, unit='strain')

    #Set up empty interferometers
    H1 = bilby.gw.detector.get_empty_interferometer("H1")

    # To analyse GW150914, we will use a 4s period duration centered on the event itself.
    # It is standard to choose the data such that it always includes a "post trigger duration" of 2s.
    # That is, there is always 2s of data after the trigger time.
    # We therefore define all times relative to the trigger time, duration and this post-trigger duration.

    # Definite times in relatation to the trigger time (time_of_event), duration and post_trigger_duration
    post_trigger_duration = 2
    duration = 4
    analysis_start = time_of_event + post_trigger_duration - duration

    H1_analysis_data = strain.crop(analysis_start , analysis_start+duration)

    H1_analysis_data.plot()
    plt.ylabel('GW Amplitude [Strain]')
    plt.savefig('figures/4seconds_around_GW150914')
    plt.show()
    #This doesn't tell us much of course! It is dominated by the low frequency noise.

    # We pass the strain data to our H1 Bilby interferometer objects.
    H1.set_strain_data_from_gwpy_timeseries(H1_analysis_data)

    # PSD
    # Parameter estimation relies on having a power spectral density (PSD)

    #We start by figuring out the amount of data needed - in this case 32 times the analysis duration.
    psd_duration = duration * 32
    psd_start_time = analysis_start - psd_duration

    H1_psd_data = strain.crop(psd_start_time , psd_start_time+psd_duration)

    # Having obtained the data to generate the PSD, we now use the standard gwpy psd method to calculate the PSD.
    # Here, the psd_alpha variable is converting the roll_off applied to the strain data into the fractional value used by gwpy.
    # This applies a window with an appropriate shape to the time-domain data.
    psd_alpha = 2 * H1.strain_data.roll_off / duration
    H1_psd = H1_psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")

    # Now that we have PSDs for H1 and L1, we can overwrite
    # the power_spectal_density attribute of our interferometers with a new PSD.

    H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=H1_psd.frequencies.value, psd_array=H1_psd.value)

    # We aren't really interested in the data at these high frequencies so
    # let's adjust the maximum frequency used in the analysis to 1024 Hz
    H1.maximum_frequency = 1024

    # CREATE A PRIOR
    # we create a prior fixing everything except the chirp mass, mass ratio,
    # phase and geocent_time parameters to fixed values.

    prior = bilby.core.prior.PriorDict()
    #prior['chirp_mass'] = UniformInComponentsChirpMass(name='chirp_mass', minimum=25.0,maximum=35.0)
    prior['chirp_mass'] = UniformInComponentsChirpMass(name='chirp_mass', minimum=15,maximum=35)
    prior['mass_ratio'] = UniformInComponentsMassRatio(name='mass_ratio', minimum=0.5, maximum=1)

    prior['phase'] = Uniform(name="phase", minimum=0, maximum=2*np.pi)
    prior['geocent_time'] = Uniform(name="geocent_time", minimum=time_of_event-0.1, maximum=time_of_event+0.1)
    prior['a_1'] =  0.0
    prior['a_2'] =  0.0
    prior['tilt_1'] =  0.0
    prior['tilt_2'] =  0.0
    prior['phi_12'] =  0.0
    prior['phi_jl'] =  0.0
    prior['dec'] =  -1.2232
    prior['ra'] =  2.19432
    prior['theta_jn'] =  1.89694
    prior['psi'] =  0.532268
    prior['luminosity_distance'] = 412.066 #https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://dcc.ligo.org/public/0182/T2200159/002/Viviana_Caceres_LIGO_SURF_Second_Interim_Report%2520%25281%2529.pdf&ved=2ahUKEwiymb3xs4iIAxXL2wIHHTyLErMQFnoECBIQAQ&usg=AOvVaw14ayIFB2DAHWE5YC2NBlQP

    print(prior)
    # For Bayesian inference, we need to evaluate the likelihood.
    # In Bilby, we create a likelihood object.
    # This is the communication interface between the sampling part of Bilby and the data.
    # Explicitly, when Bilby is sampling it only uses the parameters and log_likelihood() of
    # the likelihood object. This means the likelihood can be arbitrarily complicated
    # and the sampling part of Bilby won't mind a bit!

    #Let's create a GravitationalWaveTransient, a special inbuilt method carefully designed
    # to wrap up evaluating the likelihood of a waveform model in some data.

    # First, put our "data" created above into a list of intererometers (only H1)
    interferometers = [H1]

    # Next create a dictionary of arguments which we pass into the LALSimulation waveform - we specify the waveform approximant here
    waveform_arguments = dict(
        waveform_approximant='TaylorF2', reference_frequency=100., catch_waveform_errors=True)

    # Next, create a waveform_generator object. This wraps up some of the jobs of converting between parameters etc
    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters)

    # Finally, create our likelihood, passing in what is needed to get going
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers, waveform_generator, priors=prior,
        time_marginalization=True, phase_marginalization=True, distance_marginalization=False)
        # Cannot use marginalized likelihood for luminosity_distance: prior is fixed


    # Run the analysis

    # Now that the prior is set-up and the likelihood is set-up (with the data and the signal mode),
    # we can run the sampler to get the posterior result.
    # This function takes the likelihood and prior along with some options
    # for how to do the sampling and how to save the data.

    result_short = bilby.run_sampler(
        likelihood, prior, sampler='dynesty', outdir='shortL', label="GW150914",
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
          clean=True,)#,
        #n_effective = 5000, dlogz=3 # <- Arguments are used to make things fast - not recommended for general use

        #)

    # result_short = bilby.result.read_in_result(outdir='short', label="GW150914")
    # sample="unif", nlive=500, dlogz=3  # <- Arguments are used to make things fast - not recommended for general use

    print(result_short.priors)
    Mc = result_short.posterior["chirp_mass"].values
    q = result_short.posterior["mass_ratio"].values


    # We can then get some useful quantities such as the 90% credible interval
    lower_bound = np.quantile(Mc, 0.05)
    upper_bound = np.quantile(Mc, 0.95)
    median = np.quantile(Mc, 0.5)
    print("Mc = {} with a 90% C.I = {} -> {}".format(median, lower_bound, upper_bound))

    # We can then plot the chirp mass in a histogram adding a region to indicate the 90% C.I.
    fig, ax = plt.subplots()
    ax.hist(result_short.posterior["chirp_mass"], bins=50)
    ax.axvspan(lower_bound, upper_bound, color='C1', alpha=0.4)
    ax.axvline(median, color='C1')
    ax.set_xlabel("chirp mass")
    plt.show()

    # We can then plot the mass ratio in a histogram adding a region to indicate the 90% C.I.
    lower_bound_q = np.quantile(q, 0.05)
    upper_bound_q = np.quantile(q, 0.95)
    median_q = np.quantile(q, 0.5)
    fig, ax = plt.subplots()
    ax.hist(result_short.posterior["mass_ratio"], bins=50)
    ax.axvspan(lower_bound_q, upper_bound_q, color='C2', alpha=0.4)
    ax.axvline(median_q, color='C2')
    ax.set_xlabel("mass ratio")
    plt.show()


    result_short.plot_corner(parameters=["chirp_mass", "mass_ratio"], prior=True, save=False)
    # plt.show()

    parameters = dict(mass_1=36.2, mass_2=29.1)
    fig = result_short.plot_corner(parameters, save=False)
    plt.show()