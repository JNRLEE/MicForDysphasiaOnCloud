"""
此模型用於吞嚥聲音分類，使用滑動窗口處理音頻特徵
功能：
1. 輸入特徵維度為 [batch_size, window_size, feature_dim]
2. 使用一維卷積神經網絡提取特徵
3. 包含批量正規化和 Dropout 以防止過擬合
"""

import tensorflow as tf
from Model.base.class_config import get_num_classes

class AutoencoderModel(tf.keras.Model):
    """自編碼器模型"""
    
    def __init__(self, config: dict):
        """
        初始化模型
        
        Args:
            config: 模型配置字典
        """
        super().__init__()
        self.config = config
        
        # 確保所有必要的配置都存在
        required_configs = [
            'window_size', 'target_dim',
            'filters', 'kernel_sizes', 'pool_sizes',
            'dropout_rate', 'use_batch_norm'
        ]
        
        for cfg in required_configs:
            if cfg not in config:
                raise ValueError(f"配置中缺少必要參數: {cfg}")
        
        # 保存輸入形狀為內部變量
        self._model_input_shape = (
            config['window_size'],
            config['target_dim']
        )
        
        # CNN特徵提取器
        self.feature_extractor = tf.keras.Sequential([
            # 輸入層，處理 [batch_size, window_size, feature_dim] 形狀的數據
            tf.keras.layers.Input(shape=self._model_input_shape),
            
            # 第一個卷積塊
            tf.keras.layers.Conv1D(
                filters=config['filters'][0], 
                kernel_size=config['kernel_sizes'][0],
                activation='relu', 
                padding='same'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=config['pool_sizes'][0]),
            tf.keras.layers.Dropout(config['dropout_rate']),
            
            # 第二個卷積塊
            tf.keras.layers.Conv1D(
                filters=config['filters'][1], 
                kernel_size=config['kernel_sizes'][1], 
                activation='relu', 
                padding='same'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=config['pool_sizes'][1]),
            tf.keras.layers.Dropout(config['dropout_rate']),
            
            # 第三個卷積塊
            tf.keras.layers.Conv1D(
                filters=config['filters'][2], 
                kernel_size=config['kernel_sizes'][2], 
                activation='relu', 
                padding='same'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=config['pool_sizes'][2]),
            tf.keras.layers.Dropout(config['dropout_rate']),
            
            # 全局池化
            tf.keras.layers.GlobalAveragePooling1D(),
            
            # 分類器
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(config['dropout_rate']),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(config['dropout_rate']),
            
            tf.keras.layers.Dense(get_num_classes(), activation='softmax')
        ])

    def call(self, inputs, training=False):
        """
        前向傳播
        
        Args:
            inputs: 輸入張量，形狀為 [batch_size, window_size, feature_dim]
                   或者是一個張量的元組
            training: 是否處於訓練模式
            
        Returns:
            輸出張量，形狀為 [batch_size, num_classes]
        """
        # 如果輸入是元組，取第一�元素
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        
        # 確保輸入維度正確
        if len(inputs.shape) == 4:
            # 如果輸入是 4D，移除 channel 維度
            inputs = tf.squeeze(inputs, axis=-1)
            
        if len(inputs.shape) != 3:
            raise ValueError(
                f"輸入維度不正確，期望為3維 (batch_size, window_size, feature_dim)，"
                f"實際為{len(inputs.shape)}維，形狀為{inputs.shape}"
            )
            
        return self.feature_extractor(inputs, training=training)
        
    def build_graph(self):
        """構建計算圖，用於可視化模型結構"""
        x = tf.keras.Input(shape=self._model_input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))