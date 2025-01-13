"""
此模型用於吞嚥聲音分類
功能：
1. 使用一維卷積神經網絡處理吞嚥聲音特徵
2. 輸入特徵維度為 [batch_size, channels, window_size]，其中：
   - channels: 512 個特徵通道
   - window_size: 129，對應約 2.58 秒的音訊 (129 * 0.02s)
3. 通過多層卷積和池化提取特徵，最後使用全連接層進行分類
4. 包含批量正規化和 Dropout 以防止過擬合
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
    Flatten,
    Permute
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
        self.channels = config.get('channels', 512)      # 特徵通道數
        self.window_size = config.get('window_size', 129)  # 時間窗口大小
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.num_classes = get_num_classes()  # 從 class_config 獲取類別數
        self.class_names = get_active_class_names()  # 獲取類別名稱
        
        # 初始化模型
        self.model = None
        
        # 設置日誌
        self.logger = logging.getLogger(__name__)
        
    def build(self):
        """建構CNN分類器模型"""
        # 定義輸入層 [batch_size, channels, window_size]
        inputs = Input(shape=(self.channels, self.window_size), name='encoder_input')
        self.logger.info(f"輸入層形狀: {inputs.shape}")
        
        # 轉換維度順序為 [batch_size, window_size, channels]，以符合 Conv1D 的要求
        x = Permute((2, 1), name='permute')(inputs)
        self.logger.info(f"維度轉換後形狀: {x.shape}")
        
        # 第一個卷積塊
        x = Conv1D(64, 7, padding='same', name='conv1')(x)
        x = BatchNormalization(name='bn1')(x)
        x = Activation('relu', name='relu1')(x)
        x = MaxPooling1D(2, name='pool1')(x)
        x = Dropout(self.dropout_rate)(x)
        self.logger.info(f"第一個卷積塊輸出形狀: {x.shape}")
        
        # 第二個卷積塊
        x = Conv1D(128, 5, padding='same', name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Activation('relu', name='relu2')(x)
        x = MaxPooling1D(2, name='pool2')(x)
        x = Dropout(self.dropout_rate)(x)
        self.logger.info(f"第二個卷積塊輸出形狀: {x.shape}")
        
        # 第三個卷積塊
        x = Conv1D(256, 3, padding='same', name='conv3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = Activation('relu', name='relu3')(x)
        x = MaxPooling1D(2, name='pool3')(x)
        x = Dropout(self.dropout_rate)(x)
        self.logger.info(f"第三個卷積塊輸出形狀: {x.shape}")
        
        # 第四個卷積塊
        x = Conv1D(512, 3, padding='same', name='conv4')(x)
        x = BatchNormalization(name='bn4')(x)
        x = Activation('relu', name='relu4')(x)
        x = MaxPooling1D(2, name='pool4')(x)
        x = Dropout(self.dropout_rate)(x)
        self.logger.info(f"第四個卷積塊輸出形狀: {x.shape}")
        
        # 全局平均池化
        x = GlobalAveragePooling1D(name='gap')(x)
        self.logger.info(f"全局平均池化輸出形狀: {x.shape}")
        
        # 全連接層
        x = Dense(512, activation='relu', name='dense1')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu', name='dense2')(x)
        x = Dropout(self.dropout_rate)(x)
        self.logger.info(f"全連接層輸出形狀: {x.shape}")
        
        # 輸出層 - 使用動態類別數
        outputs = Dense(self.num_classes, activation='softmax', name='classifier_output')(x)
        self.logger.info(f"輸出層形狀: {outputs.shape}")
        
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
            x: 輸入數據，形狀為 (batch_size, channels, window_size)
            
        Returns:
            預測結果
        """
        return self.model.predict(x)