"""
此代碼實現了基於全連接層的音頻特徵分類模型
"""

import tensorflow as tf
from Model.base.class_config import get_num_classes

class WavFeatureCNN(tf.keras.Model):
    """
    簡化版的全連接層分類模型,使用較大的網絡容量和穩定的訓練策略
    """
    def __init__(self, config=None):
        """
        初始化模型,設置所有層
        """
        super().__init__()
        self.transpose = TransposeLayer()
        
        # 第一個全連接層組
        self.dense1 = tf.keras.layers.Dense(512, name='dense1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.act1 = tf.keras.layers.ReLU(name='act1')
        self.drop1 = tf.keras.layers.Dropout(0.3, name='drop1')
        
        # 第二個全連接層組
        self.dense2 = tf.keras.layers.Dense(128, name='dense2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.act2 = tf.keras.layers.ReLU(name='act2')
        self.drop2 = tf.keras.layers.Dropout(0.3, name='drop2')
        
        # 輸出層
        self.output_dense = tf.keras.layers.Dense(4, name='output')

    def call(self, inputs, training=False):
        """
        前向傳播
        """
        x = self.transpose(inputs)
        x = tf.cast(x, tf.float32)
        
        # 第一個全連接層組
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        
        # 第二個全連接層組
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.drop2(x, training=training)
        
        # 輸出層
        return self.output_dense(x)

    def get_config(self):
        return super().get_config()

    def build(self, input_shape):
        self.model.build(input_shape)
        self.built = True