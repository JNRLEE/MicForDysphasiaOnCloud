"""
三元組損失模型：使用三元組損失進行訓練的模型
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import logging
from typing import Tuple, Optional, List

from Model.base.base_model import BaseModel

class TripletLossModel(BaseModel):
    """三元組損失模型類"""
    
    def __init__(self, config):
        """初始化模型
        
        Args:
            config: 配置對象，包含模型配置
        """
        super().__init__(config)
        
        # 設置模型保存路徑
        self.model_save_dir = os.path.join(self.config.save_dir, 'triplet_loss')
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # 構建模型
        self.build()
    
    def build(self) -> None:
        """構建模型
        
        構建三元組損失模型，包括：
        1. 特徵提取器（CNN基礎網絡）
        2. 嵌入層
        3. 三元組損失層
        """
        # 定義輸入
        input_shape = (
            self.config.input_height,
            self.config.input_width,
            self.config.input_channels
        )
        anchor_input = Input(shape=input_shape, name='anchor_input')
        positive_input = Input(shape=input_shape, name='positive_input')
        negative_input = Input(shape=input_shape, name='negative_input')
        
        # 構建共享的特徵提取器
        embedding_model = self._build_embedding_model(input_shape)
        
        # 獲取三個輸入的嵌入向量
        anchor_embedding = embedding_model(anchor_input)
        positive_embedding = embedding_model(positive_input)
        negative_embedding = embedding_model(negative_input)
        
        # 添加三元組損失層
        triplet_loss_layer = Lambda(
            self._triplet_loss,
            output_shape=(1,),
            name='triplet_loss'
        )([anchor_embedding, positive_embedding, negative_embedding])
        
        # 構建完整模型
        self.model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=triplet_loss_layer,
            name='triplet_loss_model'
        )
        
        # 編譯模型
        self.model.compile(optimizer=Adam(learning_rate=self.config.learning_rate))
        
        # 保存嵌入模型以便後續使用
        self.embedding_model = embedding_model
    
    def _build_embedding_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """構建嵌入模型
        
        Args:
            input_shape: 輸入形狀 (height, width, channels)
            
        Returns:
            Model: 嵌入模型
        """
        inputs = Input(shape=input_shape)
        
        # 使用CNN基礎網絡提取特徵
        x = self._build_cnn_base(inputs)
        
        # 添加嵌入層
        embeddings = Dense(
            self.config.embedding_size,
            activation=None,
            name='embedding'
        )(x)
        
        # L2正則化
        embeddings = Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='l2_normalize'
        )(embeddings)
        
        return Model(inputs=inputs, outputs=embeddings, name='embedding_model')
    
    def _triplet_loss(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """計算三元組損失
        
        Args:
            inputs: [anchor_embedding, positive_embedding, negative_embedding]
            
        Returns:
            tf.Tensor: 損失值
        """
        anchor, positive, negative = inputs
        
        # 計算正負樣本對之間的距離
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # 基本三元組損失
        basic_loss = pos_dist - neg_dist + self.config.margin
        
        # 應用平滑
        loss = tf.maximum(basic_loss, 0.0)
        
        return loss
    
    def train(
        self,
        data_loader,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
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
        # 設置默認的回調函數
        if callbacks is None:
            callbacks = []
            
        # 添加模型檢查點
        checkpoint_path = os.path.join(
            self.model_save_dir,
            'model_checkpoint.h5'
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='loss',
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        
        # 訓練模型
        history = self.model.fit(
            x=[x_train, x_train, x_train],  # 使用相同的數據作為輸入
            y=np.zeros(len(x_train)),  # 損失在模型內部計算
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2 if x_val is None else 0.0,
            validation_data=(
                [x_val, x_val, x_val],
                np.zeros(len(x_val))
            ) if x_val is not None else None
        )
        
        return history
    
    def encode_audio_file(self, segments: np.ndarray) -> np.ndarray:
        """對音頻文件的所有片段進行編碼
        
        Args:
            segments: 音頻文件的所有LPS片段
            
        Returns:
            np.ndarray: 平均嵌入向量
        """
        # 獲取每個片段的嵌入向量
        embeddings = self.embedding_model.predict(segments)
        
        # 計算平均嵌入向量
        return np.mean(embeddings, axis=0)
    
    def predict_audio_file(self, segments: np.ndarray) -> int:
        """預測音頻文件的類別
        
        Args:
            segments: 音頻文件的所有LPS片段
            
        Returns:
            int: 預測的類別
        """
        # 獲取音頻文件的平均嵌入向量
        embedding = self.encode_audio_file(segments)
        
        # TODO: 實現最近鄰分類或其他分類方法
        # 當前返回虛擬結果
        return 0 