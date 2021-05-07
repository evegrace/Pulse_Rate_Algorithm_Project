#!/usr/bin/env python
# coding: utf-8

# # Test Your Algorithm
# 
# ## Instructions
# 1. From the **Pulse Rate Algorithm** Notebook you can do one of the following:
#    - Copy over all the **Code** section to the following Code block.
#    - Download as a Python (`.py`) and copy the code to the following Code block.
# 2. In the bottom right, click the <span style="color:blue">Test Run</span> button. 
# 
# ### Didn't Pass
# If your code didn't pass the test, go back to the previous Concept or to your local setup and continue iterating on your algorithm and try to bring your training error down before testing again.
# 
# ### Pass
# If your code passes the test, complete the following! You **must** include a screenshot of your code and the Test being **Passed**. Here is what the starter filler code looks like when the test is run and should be similar. A passed test will include in the notebook a green outline plus a box with **Test passed:** and in the Results bar at the bottom the progress bar will be at 100% plus a checkmark with **All cells passed**.
# ![Example](example.png)
# 
# 1. Take a screenshot of your code passing the test, make sure it is in the format `.png`. If not a `.png` image, you will have to edit the Markdown render the image after Step 3. Here is an example of what the `passed.png` would look like 
# 2. Upload the screenshot to the same folder or directory as this jupyter notebook.
# 3. Rename the screenshot to `passed.png` and it should show up below.
# ![Passed](passed.png)
# 4. Download this jupyter notebook as a `.pdf` file. 
# 5. Continue to Part 2 of the Project. 

# In[1]:


# replace the code below with your pulse rate algorithm.
import glob

import numpy as np
import scipy as sp
import scipy.io
import scipy.signal



def LoadTroikaDataset():
    
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls

def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]

def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability. 

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estim0tes = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))

def Evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs, confs = [], []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
        # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return AggregateErrorMetric(errs, confs)

def BandPassFilter(signal, fs):
    '''
    BandPassFilter
    
    Returns:
        Bandpassed filtered signal
        '''
    pass_band = (40/60.0 , 240/60.0)
    num, den = sp.signal.butter(3, pass_band, btype = 'bandpass', fs = fs)
    filteredSignal = sp.signal.filtfilt(num,den,signal)
    return filteredSignal


def PlotSpectrogram(signal, fs = 125):
    """PlotSpectrogram plot the spectrogram of a given signal"""
    plt.figure(figsize = (12,8))
    plt.specgram(signal, Fs = fs, NFFT = 250, noverlap = 125, xextent = [0, len(signal) / fs / 60])
    plt.xlabel('Time(min)')
    plt.ylabel('Frequency(Hz)')
    

def FourierTransform(signal, fs = 125):
    '''
    Calculates the Fourier Transform of a signal with sampling frequency fs
    Returns:
        freq and magnitude after fft'''
    freq = np.fft.rfftfreq(4*len(signal),1/fs)
    fft = np.abs(np.fft.rfft(signal, 4*len(signal)))
    return freq, fft

def EstimatePulseRate(ppg, acc, fs = 125):
    
    #band pass signals
    #ppg_fft[ppg_freqs <=40/60] = 0.0
    #ppg_fft[ppg_freqs >= 240/60] = 0.0
    
       
    #fourier transform for the ppg
    ppg_freq, ppg_fft = FourierTransform(ppg,fs)
    
    #acc_fft[acc_freqs <= 40/60] = 0.0
    #acc_fft[acc_freqs >= 240/60] = 0.0
    
    #fourier transform for the accelerometer
    acc_freq, acc_fft = FourierTransform(acc, fs)
    
    # Find dominant frequencies for the accelerometer data.
    #(Check their strongest frequencies, if they are the same,
    #do something to find the second most probable frequency inside the ppg signal)
    
    ppg_freq_max = ppg_freq[np.argmax(ppg_fft)]
    acc_freq_max = acc_freq[np.argmax(acc_fft)]
        
    if (acc_freq_max == ppg_freq_max) and len(ppg_fft) >=2:
        pulse_freq = ppg_freq[np.argsort(ppg_fft)[::-1][1]]
        pulse_rate = pulse_freq * 60
        
    else:
        pulse_freq = ppg_freq_max
        pulse_rate = pulse_freq * 60
    
    #You can answer this by summing the frequency spectrum near the pulse rate estimate 
    #and dividing it by the sum of the entire spectrum.
    spectral_energy_ppg = np.square(ppg_fft)
    window_f = 5 / 60 
    confidence = (np.sum(spectral_energy_ppg[(ppg_freq >= (pulse_freq - window_f)) & (ppg_freq <= (pulse_freq + window_f))])) / np.sum(spectral_energy_ppg)
    
    
    return pulse_rate, confidence

def RunPulseRateAlgorithm(data_fl, ref_fl):
    # Load data using LoadTroikaDataFile
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
    
    fs = 125
    
    # Bandpass Filter
    ppg = BandPassFilter(ppg, fs)
    accx = BandPassFilter(accx, fs)
    accy = BandPassFilter(accy, fs)
    accz = BandPassFilter(accz, fs)
    
    # Compute pulse rate estimates and estimation confidence.
    
    # Retrieve the reference heart rates
    bpm = sp.io.loadmat(ref_fl)['BPM0']
    
    # calculate acc from (x,y,z)
    acc = np.sqrt(accx**2 + accy**2 + accz**2)
    
    estimate_bmp = []
    confidences = []
    
    # Windowing our ppg and acc signal to estimate
    window_length_s = 8
    window_shift_s = 2
    fs = 125
    
    window_length = window_length_s * fs
    window_shift = window_shift_s * fs   
    
    
    for i in range(0, len(ppg) - window_length, window_shift):
        ppg_window = ppg[i: i + window_length]
        acc_window = acc[i: i + window_length]
        
        # get the pulse rate estimates 
        pulse_rate, confidence = EstimatePulseRate(ppg_window, acc_window, fs = fs)
        estimate_bmp.append(pulse_rate)
        confidences.append(confidence)
        
        
    # Return per-estimate mean absolute error and confidence as a 2-tuple of numpy arrays.
    errors = np.abs(np.diag(estimate_bmp - bpm))
    return errors, confidence  

