# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:43:14 2021

@author: Stacia16
"""
import os
import librosa
import numpy as np
import scipy
import glob
#from scipy import signal

from natsort import natsorted
from tqdm import tqdm

eps_64 = np.finfo(np.float64).eps
eps_32 = np.finfo(np.float32).eps

def listing(root,ext):
    
    
    list_dir=str(root)+'\*.'+ext.lower()
    file_list = glob.glob(list_dir)
    
    return file_list

def auto_list(directory,nat_sort=False):
    directory_list =[os.path.join(root,file) for root, subdirectories, files in os.walk(directory) for file in files]
    print('\nInput folder:\t{}'.format(directory))
    print('\nTotal number:\t{}\n'.format(len(directory_list)))
    if nat_sort is True:
        return natsorted(directory_list)
    else:
        return directory_list

def load_wavs(wav_list, sr):

    wavs = list()
    for file_path in tqdm(wav_list,desc='Wav Reading'):
        wav, _ = librosa.load(file_path, sr = sr, mono = True)
        #wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs

def lps_extract(wavs, frame_size = 256, overlap = 128 ,fft_size = 256, to_list = False, prevent_zero = False):
    
    fea_dim = overlap+1
    
    if isinstance(wavs, list): #如果是list就執行以下，如果不是就執行else
        lps_list, lps_info = [None]*len(wavs), [None]*len(wavs)
        
        for wav_id, wav in enumerate(wavs):
            spec = librosa.stft(wav,
                                n_fft=frame_size,
                                hop_length=overlap,
                                win_length=fft_size,
                                window=scipy.signal.windows.hamming)
            mag = abs(spec)**2
            if prevent_zero:
                mag[mag==0]=eps_64
            lps = np.log10(mag)
            phase=np.angle(spec)
            mean = np.mean(lps, axis=1).reshape(fea_dim,1)
            std = np.std(lps, axis=1).reshape(fea_dim,1)
            lps = (lps-mean)/std
            lps_list[wav_id] = lps
            lps_info[wav_id] = {'phase':phase ,'mean':mean,'std':std, 'frame_size':frame_size, 'overlap':overlap}
        if to_list:
            return lps_list,lps_info
        else:
            return np.hstack(lps_list),lps_info
    else:
        spec = librosa.stft(wavs,
                            n_fft=frame_size,
                            hop_length=overlap,
                            win_length=fft_size,
                            window=scipy.signal.windows.hamming)
        mag = abs(spec)**2 #平方
        if prevent_zero:
            mag[mag==0]=eps_64 #如果數值是零，就給接近於零的數值
        lps = np.log10(mag) #取log
        phase=np.angle(spec)
        mean = np.mean(lps, axis=1).reshape(fea_dim,1)
        std = np.std(lps, axis=1).reshape(fea_dim,1)
        lps = (lps-mean)/std
        lps_info = {'phase':phase ,'mean':mean,'std':std, 'frame_size':frame_size, 'overlap':overlap}
        return lps,lps_info

    
def lps_rebuild(lpss, info):
    
    if isinstance(lpss, list):
        wavs_list = [None]*len(lpss)
        
        for lps_id, lps in enumerate(lpss):
            
            phase = info[lps_id]['phase']
            mean, std = info[lps_id]['mean'], info[lps_id]['std']
            frame_size, overlap = info[lps_id]['frame_size'], info[lps_id]['overlap'] 
            
            # lps = (lps*std)+mean
            pow_spec = np.sqrt(np.power(10,lps))
            FrameNum = pow_spec.shape[1]
            sig = np.zeros((1,(FrameNum-1)*overlap+frame_size))
            
            Spec = pow_spec*np.exp(1j*phase)
            Spec = np.concatenate((Spec, np.flipud(np.conjugate(Spec[1:-1,:]))), axis=0)
            for i in range(0, FrameNum):
                start = i*overlap
                s = Spec[:,i]
                sig[0,start:start+frame_size] = sig[0,start:start+frame_size] + np.real(np.fft.ifft(s, frame_size))

            wavs_list[lps_id] = np.squeeze(sig)

        return wavs_list
    else:
        phase = info['phase']
        mean, std = info['mean'], info['std']
        frame_size, overlap = info['frame_size'], info['overlap'] 

        lpss = (lpss*std)+mean
        pow_spec = np.sqrt(np.power(10,lpss))
        FrameNum = pow_spec.shape[1]
        sig = np.zeros((1,(FrameNum-1)*overlap+frame_size))
        
        Spec = pow_spec*np.exp(1j*phase)
        Spec = np.concatenate((Spec, np.flipud(np.conjugate(Spec[1:-1,:]))), axis=0)
        for i in range(0, FrameNum):
            start = i*overlap
            s = Spec[:,i]
            sig[0,start:start+frame_size] = sig[0,start:start+frame_size] + np.real(np.fft.ifft(s, frame_size))
        
        single_wav = np.squeeze(sig)     
        return single_wav   

def OverlapAdd(X, phase, frame_size = 256, overlap = 128 ):
    
    FreqRes = X.shape[0]
    FrameNum = X.shape[1]
    Spec = X*np.exp(1j*phase)
    Spec = np.concatenate((Spec, np.flipud(np.conjugate(Spec[1:-1,:]))), axis=0)

    sig = np.zeros((1,(FrameNum-1)*overlap+frame_size))
    for i in range(0, FrameNum):
        start = i*overlap
        s = Spec[:,i]
        sig[0,start:start+frame_size] = sig[0,start:start+frame_size] + np.real(np.fft.ifft(s, frame_size))
    return np.squeeze(sig)

def PowerSpectrum2Wave(log10powerspectrum,yphase):
    logpowspectrum = np.log(np.power(10, log10powerspectrum)) #log power spectrum
    sig = OverlapAdd(np.sqrt(np.exp(logpowspectrum)), yphase)
    return sig
    
def logPowerSpectrum(data, frame_size = 256, overlap = 128):

    window = np.hamming(frame_size)
    phase=[]
    spec=[]
    for t in range(0, data.shape[0]-frame_size, overlap):
        seg = window*data[t:t+frame_size]
        fftspectrum = np.fft.fft(seg)
        phase.append(np.angle(fftspectrum)[0:frame_size/2+1]) 
        fftspectrum = abs(fftspectrum[0:frame_size/2+1])
        spec.append(fftspectrum)
        
    phase = np.transpose(np.array(phase))
    spec =  np.transpose(np.array(spec))**2
    spec = np.log10(spec)
    return spec, phase
    

### array control ###
def time_splite(x, time_len, padding = False):
    # time_splte_list =[]
    splite_num = (x.shape[0]-time_len)+1
    print("x: " + str(x.shape[0]) + "//" + str(x.shape[1]) + "//" + str(time_len))
    print("time_len: " + str(time_len) + " splite_num: " + str(splite_num) )
    splte_array = np.empty((splite_num,time_len,x.shape[1]))
    for i in range((x.shape[0]-time_len)+1):
        indice = slice(i,i+time_len)
        # time_splte_list.append(x[indice])
        splte_array[i,:,:] = x[indice]
    if padding:
        for j in range(time_len//2):
            top=np.insert(splte_array[:1,0:time_len-1,:],1,splte_array[:1,0,:],axis=1)
            bottom=np.insert(splte_array[-1:,1:time_len,:],-1,splte_array[-1:,-1,:],axis=1)
            splte_array = np.concatenate((top,splte_array,bottom),axis=0)
            # print(j)
    return splte_array

def time_merge(x):
    time_len = x.shape[1]
    merge_temp_mat = np.zeros([x.shape[0]+time_len-1,x.shape[2]],dtype='float32')
    time_merge_mat = np.zeros([x.shape[0]+time_len-1,x.shape[2]],dtype='float32')
    seq_list = list(x)
    
    for seq_idx,seq_batch in enumerate(seq_list): 
        temp_mat = merge_temp_mat[seq_idx:seq_idx+time_len]
        merge_temp_mat[seq_idx:seq_idx+time_len]= temp_mat + seq_batch 
    
    div_num=0   
    for time_idx,time_val in enumerate(merge_temp_mat):
        if time_idx<time_len:
            div_num = div_num+1
        elif time_idx>(len(seq_list)-1):    
            div_num = div_num-1
            
        time_merge_mat[time_idx] = time_val/div_num
            
    return time_merge_mat

    
    
def context_full(mat_data, frame_axis, context_num, concat_axis):

    context_range = [-context_num,context_num+1]
    
    Mat_frames = mat_data.shape[frame_axis]
    Mat_dims = mat_data.ndim
    
    main_list,insert_list = [None]*Mat_dims,[None]*Mat_dims
    
    assert frame_axis < Mat_dims
    
    for shift_index in range(context_range[0],context_range[1]):
        list_of_zero = [0]*abs(shift_index)
        
        if shift_index < 0:
            main_list[frame_axis] = slice(0,Mat_frames+shift_index)
            insert_list[frame_axis] = slice(0,1)
            
        elif shift_index > 0:
            main_list[frame_axis] = slice(shift_index,Mat_frames)
            insert_list[frame_axis] = slice(Mat_frames-1,Mat_frames)
        else:
            main_list[frame_axis] = slice(0,Mat_frames)
    
        for list_idx,(main_item, insert_item) in enumerate(zip(main_list,insert_list)):
            if main_item == None :
                insert_list[list_idx] = slice(None)
                
                if list_idx != frame_axis:
                    main_list[list_idx] = slice(None)
                else:
                    main_list[list_idx] = slice(0,1)
                    
        main_indices = tuple(main_list)
        insert_indices = tuple(insert_list)
        
        if shift_index == context_range[0]:
            new_data = np.insert(mat_data[main_indices],list_of_zero,mat_data[insert_indices],axis=frame_axis)
            
        elif shift_index == 0:
            prep_data = mat_data[main_indices]
            new_data = np.concatenate((new_data,prep_data),axis=concat_axis)
            
        else:
            prep_data = np.insert(mat_data[main_indices],list_of_zero, mat_data[insert_indices], axis=frame_axis)
            new_data = np.concatenate((new_data,prep_data),axis=concat_axis)
            
    return new_data
    
    
    
    