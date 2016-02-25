#! /usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Whitening of data around GW150914
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from numpy import *
import matplotlib
from matplotlib.font_manager import FontProperties
matplotlib.use("Agg")
from matplotlib.mlab import *
import pylab
from scipy import signal


# Example time-series filtering function (zero phase) - not used 
# x is the input time-series (numpy array)
# fc is the desired corner frequency of the low-pass filter (float)
# fs is the sample rate of x (float)
def lowpass(x,fc,fs):
    nyq = fs/2
    b, a = signal.butter(4, fc/nyq, 'lowpass')
    y = signal.filtfilt(b, a, x)
    return y



# Function to whiten a time-series input using a given PSD
# Calculates the FFT, divides by ASD, returns iFFT
#
# The normalization for the ffts here is not
# (in particular the whitened output is not returned in units of sigma)
#
# time_series is a numpy array of input data to be whitened
#
#    Fs       is the sample rate of time_series
#
#   PSD       is the PSD used to whiten the time-series
#
#   freq      is the frequency array corresponding to the PSD, this is used 
#             to interpolate the PSD to the precision of the FFT of time_series
#
#  low-pass   is a low-pass filter frequency (FFT is zeroed above this frequency)
#
#  high-pass  is a high-pass filter frequency (FFT is zeroed below this frequency)
#
#
# Note that returned data is normalized to units of 'sigma' away from white gaussian noise
# This depends on the sqrt of the sampling rate - 16k data will have 2x smaller excursions than 4k data
def data_condition(time_series, Fs, PSD, freq, lowpass=None, highpass=None):

    # FFT the data - normalize by the sampling rate
    dt = (1./Fs)
    transform = fft.rfft(time_series)
    transform_len = len(transform)
    df = 1./(dt*len(transform))
    
    frequencies_to_interpolate = fft.rfftfreq(len(time_series), d=dt)

    interpolated_PSD = interp(frequencies_to_interpolate, freq, PSD)

    # get the coefficients to whiten the FFT of the timeseries
    # these are the inverse of the sqrt of the interpolated PSD
    # not sure why the factor of sqrt(2) is necessary, the PSD should already be one-sided?
    coefficients = sqrt(2*dt/interpolated_PSD)

    if lowpass:
        # Replace all bins with frequencies greater than lowpass cutoff with zeros
        coefficients = where(frequencies_to_interpolate > lowpass, 0.0, coefficients)

    if highpass:
        # Replace all bins with frequencies less than highpass cutoff with zeros
        coefficients = where(frequencies_to_interpolate < highpass, 0.0, coefficients)

    # generate the whitened time series as the inverse Fourier transform of the freq-domain timeseries times the coefficients
    whitened_time_series = fft.irfft(transform*coefficients)

    return whitened_time_series




##### Plotting junk

fig_width_pt = 600  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
#golden_mean = (2.236-1.0)/2.0         # Aesthetic ratio
golden_mean = 0.6
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size = [fig_width,fig_height]

matplotlib.rcParams.update({'savefig.dpi':250,
                            'text.usetex':True,
                            'figure.figsize':fig_size,
                            'font.family':"serif",
                            'font.serif':["Times"],
                            'xtick.major.pad':'8'})

#####



# this you just have to know
event_gps = 1126259462


# Grab the strain data from the txt file, 32 seconds of data sampled at 16384Hz
H1_ht16 = genfromtxt('H-H1_LOSC_16_V1-1126259446-32.txt')

# Alternative data file with 512 seconds of data - columns are GPS time, H1_ht, L1_ht
#data = genfromtxt('GW150914_strain.txt')
#H1_ht16 = data[:,1]

# these parameters are defined in the header text of the datafile
t_start = 1126259446
Fs = 16384.0

duration = len(H1_ht16)/Fs

# build a time vetor that has the event shortly after t=0
t16 = arange(t_start,t_start+duration,1./Fs) - event_gps

# get the PSD for the full data segment with 1/6 Hz resolution
# use median estimation to be robust to glitches

# Define parameters for FFT
stride = 6.0   # FFT stride in seconds
overlap = 3.0  # overlap in seconds (50%)

H1_Pxx, freq, tH = specgram(H1_ht16, NFFT=int(stride*Fs), Fs=int(Fs), noverlap=int(overlap*Fs))

# calculate the median of the PSD and correct for the bias of the median relative to the RMS average
H1_PSD16 = median(H1_Pxx,1)/log(2)

# generate the whitened time series
H1_ht16_whiten = data_condition(H1_ht16, Fs, H1_PSD16, freq, lowpass=300, highpass=20)

# sanity check, PSD of whitened times series (should be flat)
#H1_Pxx_w16, freq, tH = specgram(H1_ht16_whiten, NFFT=int(stride*Fs), Fs=int(Fs), noverlap=int(overlap*Fs))






fignum=0

fignum+=1
pylab.figure(fignum)

ax1 = pylab.subplot(2,1,1)

pylab.plot(t16,H1_ht16,'r-',linewidth=0.8,label='H1 Raw Data')

pylab.grid(True, which='both', linestyle=':',alpha=0.8)
pylab.xlim(0.2,0.55)
pylab.ylim(-8e-19,8e-19)

pylab.xticks(visible=False)
pylab.legend(loc=3,fancybox=True,prop={'size':10})
pylab.xticks(fontsize=10)
pylab.yticks(fontsize=8)
pylab.ylabel('Strain',fontsize=12)
ax1.get_yaxis().set_label_coords(-0.05,0.5)  # this command offsets the y-axis label by a defined amount so it's the same for both plots


ax2 = pylab.subplot(2,1,2)

pylab.plot(t16,H1_ht16_whiten,'k-',linewidth=0.8,label='H1 Whitened Data (16384Hz)')

pylab.legend(loc=3,fancybox=True,prop={'size':10})
pylab.grid(True, which='both', linestyle=':',alpha=0.8)
pylab.xlim(0.2,0.55)
pylab.ylim(-2,2)

pylab.xticks(fontsize=10)
pylab.yticks(fontsize=8)
pylab.ylabel('Whitened Strain',fontsize=12)
ax2.get_yaxis().set_label_coords(-0.05,0.5)

pylab.xlabel('Time [sec] from GPS=' + str(event_gps),fontsize=12)

pylab.savefig('whitened_timeseries_GW150914.png',bbox_inches='tight')
pylab.close

