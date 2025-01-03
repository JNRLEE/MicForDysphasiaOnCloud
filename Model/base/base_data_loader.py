"""
基礎數據加載器：定義所有數據加載器的基本接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Optional, List

class BaseDataLoader(ABC):
    """基礎數據加載器類"""
    
    def __init__(self, config):
        """初始化數據加載器
        
        Args:
            config: 配置對象，包含數據集配置
        """
        self.config = config
        self.wav_data_path = config.dataset.base_paths[0]
        self.processed_data_path = os.path.join(self.wav_data_path, 'ProcessedData')
        
    @abstractmethod
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                Tuple[np.ndarray, np.ndarray], 
                                Tuple[np.ndarray, np.ndarray]]:
        """加載數據集
        
        Returns:
            Tuple: (訓練集, 驗證集, 測試集)
        """
        pass
    
    @abstractmethod
    def generate_batch(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成一個批次的數據
        
        Args:
            X: 特徵數據
            y: 標籤數據
            batch_size: 批次大小
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (批次特徵, 批次標籤)
        """
        pass 