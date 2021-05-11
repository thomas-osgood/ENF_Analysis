#!/usr/bin/python3
"""
Program Name: enf_analysis.py
Created By: Thomas Osgood
Description:
    Program designed to extract ENF traces from audio files.
"""

# Import Required Libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.signal
from datetime import datetime
import time
import tqdm

# Global Variables
enf_freq = 50
low_freq = enf_freq - 1
high_freq = enf_freq + 1

def fir_bandpass(data, fs, lowpass, highpass, usr = 1, dsr = 1):
    """
    Function Name: fir_bandpass
    Description:
        Make an FIR bandpass filter using the firwin and upfirdn
        functions from scipy.signal.
    Input(s):
        data - data to filter.
        fs - sampling rate.
        lowpass - low frequency cutoff.
        highpass - high frequency cutoff.
        usr - upsample rate for upfirdn (optional. default = 1).
        dsr - downsample rate for upfirdn (optional. default = 1).
    Return(s):
        y - filtered data.
    """
    y = np.array([])
    nyq = fs / 2
    h_nyq = nyq

    if (h_nyq % 2) == 0:
        h_nyq += 1

    h_low = lowpass / (nyq * 1.0)
    h_high = highpass / (nyq * 1.0)

    h = scipy.signal.firwin(fs+1, [h_low, h_high], pass_zero=False)
    y = scipy.signal.upfirdn(h,data)
    return y
        
def butter_bandpass(lowcut, highcut, nyq, order=None):
    """
    Function Name: butter_bandpass
    Description:
        Function to setup butterworth bandpass filter and
        return the proper coefficients.
    Input(s):
        lowcut - low cutoff frequency
        highcut - high cutoff frequency
        nyq - nyquist rate (sample_rate / 2)
        order - filter order (optional. default = 2)
    Return(s):
        b , a - filter coefficients
    """
    # Check If Optional Arg Is None
    if order is None:
        order = 2

    # Set Bandpass Frequencies
    low = lowcut / nyq
    high = highcut / nyq

    # Determine Coefficients For Filter Setup
    b, a = scipy.signal.butter(order, [low, high], btype='band')

    return b, a

def butter_bandpass_filter(data, lowcut, highcut, nyq, order=None):
    """
    Function Name: butter_bandpass_filter
    Description:
        Function to setup and filter data using a butterworth
        bandpass filter.
    Input(s):
        data - data to filter
        lowcut - low cutoff frequency
        highcut - high cutoff frequency
        nyq - nyquist rate (sample_rate / 2)
        order - order of filter (optional. default = 2)
    Return(s):
        y - filtered data
    """
    # Check If Optional Arg Is None
    if order is None: 
        order = 2

    # Get Coefficients And Filter Signal
    b, a = butter_bandpass(lowcut, highcut, nyq, order=order)
    y = scipy.signal.lfilter(b, a, data)

    # Return Filtered Data
    return y

# Main Function
def main():
    global enf_freq, low_freq, high_freq
    showFirst = False

    # Set Filename For Analysis
    filename = "pc.wav"
    #filename = "RR.wav"

    print("-"*50)
    fname_inp = input("[] Please Enter Filename [default = pc.wav]: ")

    if not(fname_inp == ""):
        filename = fname_inp

    enf_inp = input("[] Please Input ENF Frequency [default = 50]: ")

    if not(enf_inp == ""):
        enf_freq = int(enf_inp)

    harmonic = 1
    upsample_order = 5
    dnsample_order = 5

    harmonic_inp = input("[] Please Enter Desired Harmonic [default = 1]: ")

    if not(harmonic_inp == ""):
        harmonic = int(harmonic_inp)

    showFirst_inp = input("[] Show First STFT Window (y/n)? ")
    showFirst_inp = showFirst_inp.lower()

    if (showFirst_inp == "y"):
        showFirst = True
    elif ((showFirst_inp == "n") or (showFirst_inp == "")):
        showFirst = False
    else:
        print(f"[!] Incorrect Input {showFirst_inp}. Defaulting to False")
        showFirst = False

    print("-"*50)
    print(f"[+] Beginning Analysis [{filename}]")

    try:
        # Get Data & Sample Rate From File
        sr, data = wavfile.read(filename)
        data, sr = librosa.load(filename, sr=sr)
    except Exception as e:
        print("[!] Something Went Wrong Reading Audio File <{filename}> ... Exiting")
        return
    
    # Set Nyquist Rate (Sample Rate / 2)
    nyq = int(sr / 2.0)

    # Determine Time-Length And Set Axis For Plotting
    time_len = (len(data) / (sr * 1.0))
    x_ax = np.linspace(0, time_len, len(data))

    # set frame size to .2 seconds
    if time_len >= 1:
        f_size = int((len(data) / time_len) * 0.2)
    else:
        f_size = int(len(data) / 50)

    # Take FFT Of Data
    fft_data = np.fft.fft(data)
    fft_data = abs(fft_data * np.conj(fft_data)) 
    x_ax_fft = np.linspace(0, sr, len(fft_data))

    # Only Take 1st Half Of FFT Data To Avoid Mirroring
    fft_data = fft_data[:nyq]
    x_ax_fft = x_ax_fft[:nyq]

    # Plot Unfiltered Data & FFT Of Data
    plt.figure()
    plt.subplot(211)
    plt.title(f"Raw Data: {filename}")
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.plot(x_ax,data)
    plt.subplot(212)
    plt.title(f"FFT Of {filename}")
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.plot(x_ax_fft,fft_data)
    plt.tight_layout()
    plt.show()

    # Filter Data Using Bandpass With Low = 49 and High = 51 (or harmonic -- the multiplier)
    filt_data = butter_bandpass_filter(np.append(data,np.zeros(data.size * 9)), low_freq*harmonic, high_freq*harmonic, nyq, order=2)

    # Take FFT Of Filtered Data
    fft_filt_data = np.fft.fft(filt_data)
    fft_filt_data = abs(fft_filt_data * np.conj(fft_filt_data))
    x_ax_fft_f = np.linspace(0, sr, len(fft_filt_data))

    # Only Take 1st Half Of FFT To Prevent Mirroring
    fft_filt_data = fft_filt_data[:nyq]
    f_filtd_freq = np.fft.fftfreq(fft_filt_data.size, d = 2./sr)
    x_ax_fft_f = x_ax_fft_f[:nyq]
    #x_ax_fft_f = np.linspace(0, sr/2.0, f_filtd_freq.size)

    # Plot FFT Of Filtered Data
    plt.figure()
    plt.title(f"FFT Of Filtered {filename}")
    plt.ylabel("Magnitude [PSD]")
    plt.xlabel("Frequency (Hz)")
    plt.plot(x_ax_fft_f, fft_filt_data)
    plt.show()

    x_ax_us = np.linspace(0, time_len, data.size)
    
    # Plot Original & Filtered Signal On Same Plot
    plt.figure(figsize=(20,40))
    plt.subplot(211)
    plt.plot(x_ax, data, 'b')
    plt.plot(x_ax, filt_data[:data.size], 'r')
    plt.title(f"Data Comparison ({filename})")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.legend(["Original Data","Filtered Data"], loc="lower left")
    plt.subplot(212)
    plt.plot(x_ax_fft, fft_data, 'b')
    plt.plot(x_ax_fft_f, fft_filt_data[:nyq], 'r')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude [PSD]")
    plt.legend(["Original Data FFT","Filtered Data FFT"], loc="upper right")
    plt.show()
 
    # UPSAMPLED FFT ############################################

    upsample_factor = 100
    upsample_datalen = len(data) * upsample_factor
    
    fdat = np.fft.fft(data, n=upsample_datalen)
    fdat_abs = abs((fdat * np.conj(fdat)) / upsample_datalen)
    fdat_nyq = int(fdat.size / 2) + 1
    fdat_abs = fdat_abs[:fdat_nyq]
    fdat_freq = np.fft.fftfreq(fdat_abs.size, d= 2./sr)
    fdat_x = np.linspace(0,sr/2,fdat_freq.size)

    # Plot Upsampled FFT vs Original FFT
    plt.figure()
    plt.subplot(211)
    plt.title(f"Upsampled FFT: Unfiltered {filename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude [PSD]")
    plt.xlim([0,2000])
    plt.plot(fdat_x,fdat_abs)
    plt.subplot(212)
    plt.title(f"Original FFT: Unfiltered {filename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude [PSD]")
    plt.xlim([0,2000])
    plt.plot(x_ax_fft,fft_data)
    plt.tight_layout()
    plt.show()

    # RECREATE SIGNAL USING INVERSE FFT ########################

    inv_fdat = np.fft.ifft(fdat)
    inv_fdat = inv_fdat[:data.size].real

    # Display Rebuilt Signal
    plt.figure()
    plt.title("Rebuilt Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(x_ax,inv_fdat)
    plt.tight_layout()
    plt.show()
    
    fi_fdat = fir_bandpass(inv_fdat, sr, low_freq*harmonic, high_freq*harmonic)

    # STFT W/ UPSAMPLE FFT ######################################
    UPSAMPLE_FACTOR = 100
    WIN_SIZE = 4096
    HOP_SIZE = 256
    WINDOW = np.hanning(WIN_SIZE)
    N_HOPS = int(np.ceil((len(fi_fdat) * 1.0) / HOP_SIZE))
    ZEROS_NEEDED = ((HOP_SIZE * N_HOPS) + WIN_SIZE) - len(fi_fdat)
    ZEROS = np.zeros(ZEROS_NEEDED)
    fi_fdat = np.append(fi_fdat, ZEROS)

    start_i = 0
    end_i = WIN_SIZE
    max_a = 0
    max_f = 0

    enf_array = np.array([])

    if showFirst:
        print("[*] Showing First Window")
        # TEST WINOW AND FREQUENCY SPECTRUM ###################################
        win1 = fi_fdat[start_i:end_i] * WINDOW
        fft_win1_rs = abs(np.fft.fft(win1, n=(sr*100)))
        fspec = np.fft.fftfreq(fft_win1_rs.size, d = 1./(sr * 100)) / 100

        win_time = (win1.size / data.size) * time_len
        win1_x = np.linspace(0,win_time,win1.size)

        win1_nyq =int(len(fft_win1_rs)/2) 
        max_a2 = np.amax(fft_win1_rs[:win1_nyq])
        where_a2 = np.where(fft_win1_rs == max_a2)
        max_f2 = fspec[where_a2]

        plt.figure()
        plt.subplot(211)
        plt.title("Window Data")
        plt.plot(win1_x,win1)
        plt.subplot(212)
        plt.title("Upsample FFT [x100]")
        plt.plot(fspec[:win1_nyq],fft_win1_rs[:win1_nyq])
        plt.tight_layout()
        plt.show()

    print("[*] Plotting specgram")
    plt.specgram(fi_fdat, NFFT=int(sr/2), Fs=sr, noverlap=256, cmap='jet_r')
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Filtered Data [{filename}]")
    plt.show()

    # Calculate ENF ########################################################
    print("[*] Beginning ENF Trace Extraction")
    tstart = time.time()
    for i in tqdm.tqdm(range(N_HOPS),desc="[*] Extracting ENF...",ascii=False,ncols=100):
        # Set Window Start And End Variables
        start_i = i * HOP_SIZE
        end_i = start_i + WIN_SIZE

        # Apply Window Function
        win = fi_fdat[start_i:end_i] * WINDOW

        # Take FFT Of Window
        fft_win_rs = np.fft.fft(win, n=(sr*UPSAMPLE_FACTOR))
        fft_win_rs = abs((fft_win_rs * np.conj(fft_win_rs)) / (sr * UPSAMPLE_FACTOR))

        # Setup Frequency Array
        frq2 = np.fft.fftfreq(fft_win_rs.size, d = 1./(sr * UPSAMPLE_FACTOR)) / UPSAMPLE_FACTOR
        hw2 =int(len(fft_win_rs)/2) 
        
        # Determine Frequency Of Most Powerful Point
        max_a2 = np.amax(fft_win_rs[:hw2])
        where_a2 = np.where(fft_win_rs == max_a2)
        max_f2 = frq2[where_a2]

        # Append Frequency To ENF Array
        enf_array = np.append(enf_array,abs(max_f2))

    tend = time.time()
    calc_time = tend - tstart
    print(f"[*] ENF Calculation Took {calc_time} Seconds")

    # Setup X Axis
    print("[*] Setting Up Time Axis")
    enf_x = np.linspace(0,time_len,enf_array.size)

    # Plot ENF Array
    print("[*] Plotting Figure")
    plt.figure()
    plt.title(f"ENF Array [{filename}]")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(low_freq*harmonic,high_freq*harmonic)
    plt.plot(enf_x,enf_array)
    plt.show()
 
    print("-"*50)
    return

# Run Main If This File Is Not An Import
if __name__ == "__main__":
    print(f"[+] Starting Analysis: {datetime.now()}")
    main()
    print(f"[+] Analysis Over: {datetime.now()}")
