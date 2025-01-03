"""
工具函數模組：提供各種通用的工具函數
包含：
1. 音頻處理相關函數
2. 特徵提取相關函數
3. 數據增強相關函數
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional
import os

def load_wavs(file_paths: List[str], sr: int) -> np.ndarray:
    """載入多個WAV文件並重採樣"""
    waves = []
    for path in file_paths:
        try:
            print(f"  正在加載: {os.path.basename(path)}")
            wav, orig_sr = librosa.load(path, sr=sr)
            print(f"    原始採樣率: {orig_sr}")
            print(f"    目標採樣率: {sr}")
            print(f"    音頻長度: {len(wav)} 採樣點 ({len(wav)/sr:.2f} 秒)")
            waves.append(wav)
        except Exception as e:
            print(f"    錯誤加載 {os.path.basename(path)}: {str(e)}")
            print(f"    詳細錯誤信息:", e)
            continue
    
    if not waves:
        print("警告：沒有成功加載任何音頻文件")
        return np.array([])
    
    # 檢查所有音頻長度是否一致
    lengths = [len(w) for w in waves]
    if len(set(lengths)) > 1:
        print(f"警告：音頻長度不一致 {lengths}")
        # 使用最短的長度
        min_len = min(lengths)
        waves = [w[:min_len] for w in waves]
    
    return np.array(waves)

def lps_extract(waves: np.ndarray, frame_size: int, overlap: int, 
                fft_size: int, to_list: bool = False) -> Tuple[np.ndarray, Optional[List]]:
    """提取對數功率譜特徵"""
    features = []
    for wav in waves:
        # 計算STFT
        stft = librosa.stft(
            wav,
            n_fft=fft_size,
            hop_length=frame_size-overlap,
            win_length=frame_size
        )
        # 計算功率譜
        power_spec = np.abs(stft) ** 2
        # 計算對數功率譜
        log_power_spec = librosa.power_to_db(power_spec, ref=np.max)
        features.append(log_power_spec)
    
    features = np.array(features)
    if to_list:
        return features, features.tolist()
    return features, None

def time_splite(features: np.ndarray, time_len: int, padding: bool = False) -> np.ndarray:
    """將特徵按時間長度分割"""
    if padding and features.shape[1] % time_len != 0:
        pad_len = time_len - (features.shape[1] % time_len)
        features = np.pad(features, ((0, 0), (0, pad_len)), 'constant')
    
    segments = []
    for i in range(0, features.shape[1] - time_len + 1):
        segment = features[:, i:i + time_len]
        segments.append(segment)
    
    return np.array(segments)

def add_noise(features: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
    """添加高斯噪聲進行數據增強"""
    noise = np.random.normal(0, 1, features.shape)
    return features + noise_factor * noise

def time_shift(features: np.ndarray, shift_range: int = 20) -> np.ndarray:
    """時間軸平移進行數據增強"""
    shift = np.random.randint(-shift_range, shift_range)
    return np.roll(features, shift, axis=1)

def freq_mask(features: np.ndarray, num_masks: int = 1, 
              mask_size: int = 20) -> np.ndarray:
    """頻率遮罩進行數據增強"""
    masked = features.copy()
    for _ in range(num_masks):
        freq_start = np.random.randint(0, features.shape[0] - mask_size)
        masked[freq_start:freq_start + mask_size, :] = 0
    return masked 