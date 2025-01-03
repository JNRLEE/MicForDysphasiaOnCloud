"""
基礎模型：定義所有模型的基本接口
"""

from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import Optional, List, Tuple
import os
import logging

class BaseModel(ABC):
    """基礎模型類"""
    
    def __init__(self, config):
        """初始化模型
        
        Args:
            config: 配置對象，包含模型配置
        """
        self.config = config
        self.model = None
        
        # 設置模型保存路徑
        self.model_save_dir = os.path.join(config.save_dir, 'models')
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    @abstractmethod
    def build(self) -> None:
        """構建模型"""
        pass
    
    @abstractmethod
    def train(
        self,
        data_loader,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        callbacks: Optional[List] = None
    ) -> tf.keras.callbacks.History:
        """訓練模型
        
        Args:
            data_loader: 數據加載器
            x_train: 訓練數據
            y_train: 訓練標籤
            x_val: 驗證數據
            y_val: 驗證標籤
            batch_size: 批次大小
            epochs: 訓練輪數
            callbacks: 回調函數列表
            
        Returns:
            History: 訓練歷史
        """
        pass
    
    def save(self, filepath: str) -> None:
        """保存模型
        
        Args:
            filepath: 保存路徑
        """
        self.model.save(filepath)
        logging.info(f"模型已保存至: {filepath}")
    
    def load(self, filepath: str) -> None:
        """加載模型
        
        Args:
            filepath: 模型文件路徑
        """
        self.model = tf.keras.models.load_model(filepath)
        logging.info(f"已加載模型: {filepath}")
    
    def _build_cnn_base(self, inputs: tf.Tensor) -> tf.Tensor:
        """構建基礎CNN網絡
        
        Args:
            inputs: 輸入張量
            
        Returns:
            tf.Tensor: 特徵張量
        """
        x = inputs
        
        for i, (f, k, d) in enumerate(zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilation_rates
        )):
            x = tf.keras.layers.Conv2D(
                filters=f,
                kernel_size=k,
                padding='same',
                dilation_rate=d,
                activation='relu',
                name=f'conv_{i}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'bn_{i}')(x)
            x = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                padding='same',
                name=f'pool_{i}'
            )(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        if self.config.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.config.dropout_rate, name='dropout')(x)
        
        return x 