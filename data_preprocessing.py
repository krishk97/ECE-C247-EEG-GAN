import numpy as np
import random
from sklearn.linear_model import LinearRegression
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

'''
This file contains helpful functions to preprocess data, such as cropping, subsampling, continuous wavelet transform, fourier transform.  
'''


def crop_data_sequentially(X,y,person, sample_len = 750, stride = 100): 
    '''
    Function that creates cropped data sequentially using given length and stride 
    INPUTS: 
    X - EEG data (num_trials, num_eeg_electrodes, time_bins,1) 
    y - target data corresponding to correct motor imagery task (num_trials, )
    person - subject performing task, ranging from 0-8 (num_trials, )
    sample_len - length of cropped sample
    stride - distance between crops 
    OUTPUT:
    X_cropped - cropped EEG data (num_samples, num_eeg_electrodes, time_bins) with 
                num_samples_per_trial = (num_time_bins - sample_len)/stride + 1
                num_samples = num_trials * num_samples_per_trial
    y_cropped - cropped target data (num_samples, )
    person_cropped - subject performing task from cropped data, ranging from 0-8 (num_samples, )
    '''   
    
    # helpful params
    num_trials,num_eeg_electrodes,num_time_bins,_ = X.shape
    num_samples_per_trial = int((num_time_bins-sample_len)/stride + 1)
    num_samples = int(num_trials*((num_time_bins-sample_len)/stride + 1))

    X_cropped = np.zeros((num_samples,num_eeg_electrodes,sample_len,1))
    y_cropped = np.zeros((num_samples,))
    person_cropped = np.zeros((num_samples,))

    for i,trial in enumerate(X):
        strt_idx = i*num_samples_per_trial
        crop = [trial[:,x*stride:x*stride+sample_len,:] for x in range(num_samples_per_trial)]
        X_cropped[strt_idx:strt_idx+num_samples_per_trial]= crop
        y_cropped[strt_idx:strt_idx+num_samples_per_trial] = [y[i]]*num_samples_per_trial
        person_cropped[strt_idx:strt_idx+num_samples_per_trial] = [person[i]]*num_samples_per_trial
    
    return X_cropped, y_cropped, person_cropped


def crop_data_random(X,y,person,sample_len=750,min_start_ind=0,num_crops=3,seed=None): 
    '''
    Function that creates cropped data from random starting indicies 
    INPUTS: 
    X - EEG data (num_trials, num_eeg_electrodes, time_bins,1) 
    y - target data corresponding to correct motor imagery task (num_trials, )
    person - subject performing task, ranging from 0-8 (num_trials, )
    sample_len - length of cropped sample
    min_start_ind - minimum starting index for crop to begin at 
    num_crops - number of crops to be used per trial 
    seed - seed to be used for random cropping
    OUTPUT:
    X_cropped - cropped EEG data (num_trials*num_crops, num_eeg_electrodes, time_bins) with er_trial
    y_cropped - cropped target data (num_trials*num_crops, )
    person_cropped - subject performing task from cropped data, ranging from 0-8 (num_trials*num_crops, )
    '''     
    if seed != None: 
        random.seed(seed)
    
    # helpful params
    num_trials,num_eeg_electrodes,num_time_bins,_ = X.shape
    
    X_cropped = np.zeros((num_trials*num_crops,num_eeg_electrodes,sample_len,1))
    y_cropped = np.zeros((num_trials*num_crops,))
    person_cropped = np.zeros((num_trials*num_crops,))

    for i,trial in enumerate(X):            
        strt_idxs = np.random.uniform(min_start_ind,num_time_bins-sample_len,num_crops)
        strt_idxs = strt_idxs.astype(int)
        crop = [trial[:,strt_idxs[k]:strt_idxs[k]+sample_len,:] for k in range(num_crops)]
        X_cropped[i*num_crops:i*num_crops+num_crops]= crop
        y_cropped[i*num_crops:i*num_crops+num_crops] = [y[i]]*num_crops
        person_cropped[i*num_crops:i*num_crops+num_crops] = [person[i]]*num_crops
    
    return X_cropped, y_cropped, person_cropped

def morlet_wavelet_transform(X,fs=250,freq_range=(1,15),freq_bins=100,w=5):
    ''' 
    Discrete continous wavelet transform of eeg data convolved with complex morlet wavelet
    INPUTS:
    X - EEG data (num_trials, num_eeg_electrodes, time_bins,1)
    fs - sampling rate in Hz
    freq_range - tuple containing min and max freq range to perform analysis within
    freq_bins - number of points between freq range being analyzed 
    w - Omega0 for complex morlet wavelet
    OUTPUTS: 
    X_cwt - Wavlet transformed eeg data (num_trials, num_eeg_electrodes,freq_bins,time_bins)
    '''
    
    N_trials,N_eegs,time_bins,_ = X.shape
    
    # values for cwt
    freq = np.linspace(freq_range[0],freq_range[1],freq_bins)
    widths = w * fs / (2 * freq * np.pi) 
    X_cwt = np.zeros((N_trials,N_eegs,widths.shape[0],time_bins))
    
    print('Performing discrete CWT convolutions...')
    for trial in tqdm_notebook(range(N_trials), desc='Trials'):
        for eeg in tqdm_notebook(range(N_eegs), desc='EEG Channel', leave=False): 
            X_cwt[trial,eeg,:,:] = np.abs(signal.cwt(np.squeeze(X[trial,eeg,:,]),signal.morlet2,widths,w=w))

    return X_cwt

# subsampling eeg data function  
def subsample_data(X,y,person,sample_every=3):
    '''
    Subsamples the data to create more trials. Used as a data augmentation method.
    Returns trials along axis=0 with truncated time length based on sampling rate,
    the modified task and person labels

    Inputs:
    X: Input data of size (trials, eegs, timebins, 1)
    y: labels of input data specifying task number of size(trials,)
    person: labels of input data specifying subject number
    sample_every: Integer specifying sampling rate over time bins
    '''
    num_trials,num_eegs,num_time_bins,_ = X.shape
    assert num_time_bins%sample_every==0, 'Time length not divisible by sampling rate'
    
    #allocate size of output arrays
    X_subsample = np.empty((int(num_trials*sample_every),num_eegs,num_time_bins//sample_every,1))
    y_subsample = np.empty(int(num_trials*sample_every),)
    person_subsample = np.empty(int(num_trials*sample_every),)

    for i in range(sample_every):
        strt_idx = int(i*num_trials)
        X_subsample[strt_idx:strt_idx+num_trials] = X[:,:,i::sample_every,:]
        y_subsample[strt_idx:strt_idx+num_trials] = np.squeeze(y)
        person_subsample[strt_idx:strt_idx+num_trials] = np.squeeze(person)

    return X_subsample,y_subsample, person_subsample

def stft_data(X,window_size=64,stride=24,freq=(0,30),draw=False):
    ''' 
    Short-time-Fourier transform (STFT) function
    INPUTS:
    X - EEG data (num_trials, num_eeg_electrodes, time_bins,1)
    window_size - fixed # of time bins to analyze frequency components within
    stride - distance between starting position of adjacent windows
    freq - frequency range to obtain spectral power components
    OUTPUTS: 
    X_stft - STFT transformed eeg data (num_trials, num_eeg_electrodes*freq_bins,time_bins,1)
    num_freq - number of frequency bins
    num_time - number of time bins
    '''
    fs = 250
    num_trials, num_eegs, N,_ = X.shape
    assert (N-window_size)/stride%1==0,'Window size and stride not valid for length of data'

    f, t, Zxx = signal.stft(X[0,0,:,:], fs=fs, axis=-2,nperseg=window_size,noverlap=window_size-stride)

    wanted_freq = np.where(np.logical_and(f>=freq[0],f<=freq[1]))[0]
    num_freq = wanted_freq.shape[0]
    num_time = t.shape[0]

    X_stft = np.empty((int(num_trials), int(num_freq*num_eegs),int(num_time),1))
    for i in range(num_trials):
        #for j in range(num_eegs):
        f, t, Zxx = signal.stft(X[i,:,:,:], fs=fs, axis=-2,nperseg=window_size,noverlap=window_size-stride)
        wanted_Zxx = Zxx[:,wanted_freq,:,:]
        wanted_Zxx = np.reshape(np.transpose(wanted_Zxx,(0,1,3,2)),(num_freq*num_eegs,num_time,1))
        if draw==True:
            draw_f = np.repeat(np.expand_dims(f[wanted_freq],axis=1),num_eegs,axis=1).T.flatten()
            plt.pcolormesh(t, range(draw_f.shape[0]), np.abs(np.squeeze(wanted_Zxx)))
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
        X_stft[i] = wanted_Zxx

    return X_stft, num_freq, num_time