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
from scipy import constants
from makewaveform import *
from optimalSNR import *
from scipy import signal

def lowpass(x,fc,fs):
    nyq = fs/2
    b, a = signal.butter(4, fc/nyq, 'lowpass')
    y = signal.filtfilt(b, a, x)
    return y


def data_condition(time_series, Fs, PSD, freq, lowpass=0, highpass=0):

    dt = (1./Fs)
    transform = dt * fft.rfft(time_series)
    transform_len = len(transform)
    df = 1./(dt*len(transform))
    
    frequencies_to_interpolate = fft.rfftfreq(len(time_series), d=dt)

    interpolated_PSD = interp(frequencies_to_interpolate, freq, PSD)

    # get the coefficients to whiten the FFT of the timeseries
    # these are the inverse of the sqrt of the interpolated PSD
    coefficients = sqrt((2.0)/interpolated_PSD)

    if lowpass:
        # Replace all bins with frequencies greater than lowpass cutoff with zeros
        coefficients = where(frequencies_to_interpolate > lowpass, 0.0, coefficients)

    if highpass:
        # Replace all bins with frequencies less than highpass cutoff with zeros
        coefficients = where(frequencies_to_interpolate < highpass, 0.0, coefficients)

    # generate the whitened time series as the inverse Fourier transform of the freq-domain timeseries times the coefficients
    whitened_time_series = real( 2.0 * df * fft.irfft(transform*coefficients) )

    return whitened_time_series





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


# Strain data
data = genfromtxt('GW150914_strain.txt')
H1_ht = data[:,0]
L1_ht = data[:,1]

event_time = 1126259462.39
event_gps = 1126259462
offset = -0.0073

duration = 256
t_start = event_gps - duration/2
Fs = 16384

t = arange(t_start,t_start+duration,1./Fs)

tidx = argmin(abs(t-event_gps))
# seconds to plot around event
sidx = 16384*3

s1 = tidx-sidx/2
s2 = tidx+sidx/2

# indices for coherent combination
tidx = argmin(abs(t-event_gps-offset))
b1 = tidx-sidx/2
b2 = tidx+sidx/2


# get the H1, L1 PSDs for the full data segment with 1/6 Hz resolution

# Define parameters for FFT
stride = 6.0   # FFT stride in seconds
overlap = 3.0  # overlap in seconds (50%)

H1_Pxx, freq, tH = specgram(H1_ht, NFFT=int(stride*Fs), Fs=int(Fs), noverlap=int(overlap*Fs))
L1_Pxx, freq, tL = specgram(L1_ht, NFFT=int(stride*Fs), Fs=int(Fs), noverlap=int(overlap*Fs))

H1_PSD = median(H1_Pxx,1)/log(2)
L1_PSD = median(L1_Pxx,1)/log(2)

H1_ht_whiten = data_condition(H1_ht, Fs, H1_PSD, freq, lowpass=300, highpass=35)
L1_ht_whiten = data_condition(L1_ht, Fs, L1_PSD, freq, lowpass=300, highpass=35)

H1_Pxx_w, freq, tH = specgram(H1_ht_whiten, NFFT=int(stride*Fs), Fs=int(Fs), noverlap=int(overlap*Fs))
H1_PSD_w = median(H1_Pxx_w,1)/log(2)

L1_Pxx_w, freq, tL = specgram(L1_ht_whiten, NFFT=int(stride*Fs), Fs=int(Fs), noverlap=int(overlap*Fs))
L1_PSD_w = median(L1_Pxx_w,1)/log(2)

H1_ht_plot = H1_ht_whiten
L1_ht_plot = -1*L1_ht_whiten





fignum=0

fignum+=1
pylab.figure(fignum)

ax1 = pylab.subplot(2,1,1)

pylab.plot(t[s1:s2]-event_gps,H1_ht_plot[s1:s2],'r-',linewidth=0.8,label='H1 Data')
pylab.plot(t[s1:s2]-event_gps-offset,L1_ht_plot[s1:s2],'b-',linewidth=0.8,label='L1 Data (shifted 7.3msec, inverted)')

pylab.grid(True, which='both', linestyle=':',alpha=0.8)
pylab.xlim(0.2,0.55)

pylab.xticks(visible=False)
pylab.legend(loc=3,fancybox=True,prop={'size':10})
pylab.xticks(fontsize=10)
pylab.yticks(fontsize=8)
pylab.ylabel('Whitened Strain',fontsize=12)
ax1.get_yaxis().set_label_coords(-0.10,0.5)


ax2 = pylab.subplot(2,1,2)

pylab.plot(t[s1:s2]-event_gps,H1_ht_plot[s1:s2]+L1_ht_plot[b1:b2],'k-',linewidth=0.8,label='Coherent Data Sum')

pylab.legend(loc=3,fancybox=True,prop={'size':10})
pylab.grid(True, which='both', linestyle=':',alpha=0.8)
pylab.xlim(0.2,0.55)

pylab.xticks(fontsize=10)
pylab.yticks(fontsize=8)
pylab.ylabel('Whitened Strain',fontsize=12)
ax2.get_yaxis().set_label_coords(-0.10,0.5)

pylab.xlabel('Time [sec] from GPS=' + str(event_gps),fontsize=12)

pylab.savefig('whitened_timeseries_GW150914.png',bbox_inches='tight')
pylab.close

