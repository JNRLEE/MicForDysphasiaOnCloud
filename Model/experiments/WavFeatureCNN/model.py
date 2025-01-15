"""
此代碼實現了基於1D CNN的音頻特徵分類模型，
使用一維卷積來捕捉時間序列特徵。
"""

import tensorflow as tf
from Model.base.class_config import get_num_classes

class AutoencoderModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 定義所有層
        self.layers_list = []
        
        # 第一個卷積塊
        self.layers_list.extend([
            tf.keras.layers.Conv1D(32, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling1D(4)
        ])
        
        # 第二個卷積塊
        self.layers_list.extend([
            tf.keras.layers.Conv1D(64, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling1D(4)
        ])
        
        # 第三個卷積塊
        self.layers_list.extend([
            tf.keras.layers.Conv1D(128, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling1D(4)
        ])
        
        # 全局池化和分類
        self.layers_list.extend([
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(get_num_classes(), activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        x = inputs
        
        # 依序通過每一層
        for layer in self.layers_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or \
               isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        return x
    
    def build_model(self, input_shape):
        """構建模型
        
        Args:
            input_shape: 輸入形狀，例如 (None, 2000, 512)
        """
        # 創建一個示例輸入來構建模型
        inputs = tf.keras.Input(shape=input_shape[1:])
        outputs = self.call(inputs)
        
        # 創建函數式模型
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # 設置模型已構建標誌
        self.built = True
        
        return self.model
    
    def summary(self, *args, **kwargs):
        """顯示模型摘要"""
        if hasattr(self, 'model'):
            return self.model.summary(*args, **kwargs)
        else:
            raise ValueError("模型尚未構建，請先調用 build_model()")