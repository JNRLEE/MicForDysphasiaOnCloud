"""
此代碼實現了基於全連接層的音頻特徵分類模型
"""

import tensorflow as tf

class WavFeatureCNN(tf.keras.Model):
    """
    極簡版的全連接層分類模型，使用最少的參數
    """
    def __init__(self, config=None):
        super(WavFeatureCNN, self).__init__()
        self.model = tf.keras.Sequential([
            # 輸入層
            tf.keras.layers.Input(shape=(512, 2000)),
            
            # 展平層
            tf.keras.layers.Reshape((512 * 2000,)),
            
            # 第一個全連接層
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.3),
            
            # 第二個全連接層
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.3),
            
            # 輸出層
            tf.keras.layers.Dense(4)
        ])

    def call(self, inputs):
        return self.model(inputs)

    def build(self, input_shape):
        self.model.build(input_shape)
        self.built = True