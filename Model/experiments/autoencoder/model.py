"""
自編碼器模型：用於特徵提取和分類
功能：
1. 將輸入數據編碼為低維表示
2. 從低維表示重建原始數據
3. 基於低維表示進行多分類
4. 支持跳躍連接以保留細節信息
5. 支持可變維度輸入
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling1D,
    Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from base.class_config import get_num_classes, get_active_class_names

class AutoencoderModel:
    def __init__(self, config: Dict):
        """初始化模型
        
        Args:
            config: 配置字典，包含模型參數
        """
        self.time_steps = config.get('time_steps', 100)  # k-means 壓縮後的維度
        self.feature_dim = config.get('feature_dim', 1)  # 單一特徵維度
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.num_classes = get_num_classes()  # 從 class_config 獲取類別數
        self.class_names = get_active_class_names()  # 獲取類別名稱
        
        # 初始化模型
        self.model = None
        
    def build(self):
        """建構CNN分類器模型"""
        # 定義輸入
        inputs = Input(shape=(self.time_steps, self.feature_dim), name='encoder_input')
        
        # 第一個卷積塊
        x = Conv1D(32, 5, padding='same', name='conv1')(inputs)
        x = BatchNormalization(name='bn1')(x)
        x = Activation('relu', name='relu1')(x)
        x = MaxPooling1D(2, name='pool1')(x)
        
        # 第二個卷積塊
        x = Conv1D(64, 5, padding='same', name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Activation('relu', name='relu2')(x)
        x = MaxPooling1D(2, name='pool2')(x)
        
        # 第三個卷積塊
        x = Conv1D(128, 5, padding='same', name='conv3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = Activation('relu', name='relu3')(x)
        x = MaxPooling1D(2, name='pool3')(x)
        
        # 全局平均池化
        x = GlobalAveragePooling1D(name='gap')(x)
        
        # 全連接層
        x = Dense(128, activation='relu', name='dense1')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='relu', name='dense2')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # 輸出層 - 使用動態類別數
        outputs = Dense(self.num_classes, activation='softmax', name='classifier_output')(x)
        
        # 創建模型
        self.model = Model(inputs=inputs, outputs=outputs, name='cnn_classifier')
        
        # 編譯模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 打印模型摘要
        self.model.summary()
        
        # 打印類別信息
        print("\n類別配置:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {i}: {class_name}")
    
    def fit(self, *args, **kwargs):
        """訓練模型的包裝函數"""
        return self.model.fit(*args, **kwargs)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """預測函數
        
        Args:
            x: 輸入數據
            
        Returns:
            預測結果
        """
        return self.model.predict(x)