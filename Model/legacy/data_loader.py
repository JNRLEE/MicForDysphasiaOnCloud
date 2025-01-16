"""
數據加載器：負責加載和處理數據
"""

import numpy as np
import os
from typing import Tuple, List, Dict
import logging
from glob import glob
import random

class TripletDataLoader:
    """三元組數據加載器類"""
    
    def __init__(self, config):
        """初始化數據加載器
        
        Args:
            config: 配置對象，包含數據加載配置
        """
        self.config = config
        self.data_dir = config.data_dir
        
        # 存儲每個音頻文件的LPS片段索引
        self.audio_file_indices = {}
        
        # 加載數據
        self._load_npy_files()
    
    def _load_npy_files(self) -> None:
        """加載所有NPY文件
        
        加載所有LPS特徵文件，並記錄每個音頻文件的片段索引
        """
        self.data = []
        self.labels = []
        
        # 遍歷所有類別目錄
        for class_dir in glob(os.path.join(self.data_dir, '*')):
            if not os.path.isdir(class_dir):
                continue
                
            class_label = int(os.path.basename(class_dir))
            
            # 遍歷該類別下的所有NPY文件
            for npy_file in glob(os.path.join(class_dir, '*.npy')):
                # 加載LPS特徵
                lps_features = np.load(npy_file)
                
                # 記錄當前音頻文件的片段索引範圍
                start_idx = len(self.data)
                self.data.extend(lps_features)
                self.labels.extend([class_label] * len(lps_features))
                end_idx = len(self.data)
                
                # 存儲音頻文件名和對應的片段索引
                audio_file = os.path.splitext(os.path.basename(npy_file))[0]
                self.audio_file_indices[audio_file] = {
                    'indices': list(range(start_idx, end_idx)),
                    'label': class_label
                }
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        logging.info(f"已加載 {len(self.data)} 個LPS片段")
        logging.info(f"已加載 {len(self.audio_file_indices)} 個音頻文件")
    
    def generate_triplets(self, num_triplets: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成三元組樣本
        
        從數據集中生成三元組(anchor, positive, negative)用於訓練。
        - anchor: 錨點樣本
        - positive: 與anchor屬於同一音頻文件的樣本
        - negative: 與anchor屬於不同類別的樣本
        
        Args:
            num_triplets: 需要生成的三元組數量
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (anchors, positives, negatives)三元組數據
        """
        anchors = []
        positives = []
        negatives = []
        
        # 獲取所有音頻文件列表
        audio_files = list(self.audio_file_indices.keys())
        
        for _ in range(num_triplets):
            # 隨機選擇一個音頻文件作為anchor
            anchor_file = random.choice(audio_files)
            anchor_info = self.audio_file_indices[anchor_file]
            
            # 從該音頻文件的片段中隨機選擇兩個不同的片段作為anchor和positive
            anchor_idx, positive_idx = random.sample(anchor_info['indices'], 2)
            
            # 選擇一個不同類別的音頻文件
            negative_files = [
                f for f in audio_files 
                if self.audio_file_indices[f]['label'] != anchor_info['label']
            ]
            negative_file = random.choice(negative_files)
            
            # 從negative文件的片段中隨機選擇一個作為negative
            negative_idx = random.choice(self.audio_file_indices[negative_file]['indices'])
            
            anchors.append(self.data[anchor_idx])
            positives.append(self.data[positive_idx])
            negatives.append(self.data[negative_idx])
        
        return (
            np.array(anchors),
            np.array(positives),
            np.array(negatives)
        )
    
    def get_audio_file_segments(self, audio_file: str) -> np.ndarray:
        """獲取指定音頻文件的所有LPS片段
        
        Args:
            audio_file: 音頻文件名(不含擴展名)
            
        Returns:
            np.ndarray: 該音頻文件的所有LPS片段
        """
        if audio_file not in self.audio_file_indices:
            raise ValueError(f"未找到音頻文件: {audio_file}")
            
        indices = self.audio_file_indices[audio_file]['indices']
        return self.data[indices] 