"""
基礎數據集類：提供數據集的基本功能
主要功能：
1. 提供數據集的基本接口
2. 定義數據集的基本屬性
3. 提供數據集的基本操作方法
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseDataset(ABC):
    """基礎數據集類"""
    
    @abstractmethod
    def process_wav_files(self) -> Tuple[np.ndarray, np.ndarray]:
        """處理音頻文件"""
        pass
    
    @abstractmethod
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                  np.ndarray, np.ndarray, np.ndarray]:
        """準備數據"""
        pass 