"""
此代碼實現了基於全連接層的音頻特徵分類模型
"""

import tensorflow as tf
from Model.base.class_config import get_num_classes

class TransposeLayer(tf.keras.layers.Layer):
    """轉置層，用於調整輸入特徵的維度順序"""
    def call(self, inputs):
        return tf.transpose(inputs, [0, 2, 1])

class WavFeatureCNN(tf.keras.Model):
    """
    簡化版的全連接層分類模型,使用較小的網絡容量和穩定的訓練策略
    """
    def __init__(self, config=None):
        """
        初始化模型,設置所有層
        """
        super().__init__()
        self.transpose = TransposeLayer()
        self.flatten = tf.keras.layers.Flatten()
        
        # 第一個全連接層組
        self.dense1 = tf.keras.layers.Dense(64, name='dense1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.act1 = tf.keras.layers.ReLU(name='act1')
        self.drop1 = tf.keras.layers.Dropout(0.3, name='drop1')
        
        # 輸出層
        num_classes = get_num_classes()  # 從配置中獲取類別數
        self.output_dense = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')

    def call(self, inputs, training=False):
        """
        前向傳播
        """
        x = tf.cast(inputs, tf.float32)
        x = self.transpose(x)
        x = self.flatten(x)
        
        # 第一個全連接層組
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        
        # 輸出層
        return self.output_dense(x)

    def get_config(self):
        return super().get_config()