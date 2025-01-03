"""
自編碼器模型：用於特徵提取和降維
功能：
1. 將輸入數據編碼為低維表示
2. 從低維表示重建原始數據
3. 支持跳躍連接以保留細節信息
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import List, Dict, Optional, Tuple
from Model.base.config import ModelConfig

class AutoencoderModel:
    def __init__(self, config: ModelConfig):
        """初始化自編碼器模型
        
        Args:
            config: ModelConfig 對象，包含模型配置
        """
        self.config = config
        self._validate_config()
        
        # 設置模型參數
        self.input_shape = config.input_shape
        self.latent_dim = config.latent_dim
        self.filters = config.filters
        self.kernel_sizes = config.kernel_sizes
        self.dilation_rates = config.dilation_rates
        self.use_skip_connections = config.use_skip_connections
        
        # 計算和驗證網絡深度
        self.network_depth = len(self.filters)
        min_dimension = min(self.input_shape[0], self.input_shape[1])
        max_depth = int(np.log2(min_dimension))
        if self.network_depth > max_depth:
            raise ValueError(
                f"網絡深度 ({self.network_depth}) 太大，"
                f"對於輸入大小 {self.input_shape}，最大允許深度為 {max_depth}"
            )
        
        # 初始化模型
        self.model = None
        self.encoder = None
        self.decoder = None
    
    def _validate_config(self):
        """驗證配置參數的完整性和有效性"""
        # 檢查必要參數
        if self.config.latent_dim is None:
            raise ValueError("必須指定 latent_dim")
        if self.config.filters is None:
            raise ValueError("必須指定 filters")
        if self.config.kernel_sizes is None:
            raise ValueError("必須指定 kernel_sizes")
        if self.config.dilation_rates is None:
            raise ValueError("必須指定 dilation_rates")
            
        # 檢查參數的一致性
        if len(self.config.filters) != len(self.config.kernel_sizes):
            raise ValueError(
                f"filters ({len(self.config.filters)}) 和 "
                f"kernel_sizes ({len(self.config.kernel_sizes)}) 的長度必須相同"
            )
        if len(self.config.filters) != len(self.config.dilation_rates):
            raise ValueError(
                f"filters ({len(self.config.filters)}) 和 "
                f"dilation_rates ({len(self.config.dilation_rates)}) 的長度必須相同"
            )
            
        # 檢查參數的有效性
        if self.config.latent_dim <= 0:
            raise ValueError(f"latent_dim ({self.config.latent_dim}) 必須大於0")
        if not all(f > 0 for f in self.config.filters):
            raise ValueError("所有的 filters 值必須大於0")
        if not all(all(k > 0 for k in ks) for ks in self.config.kernel_sizes):
            raise ValueError("所有的 kernel_size 值必須大於0")
        if not all(d > 0 for d in self.config.dilation_rates):
            raise ValueError("所有的 dilation_rate 值必須大於0")
    
    def _build_encoder_block(
        self,
        inputs: tf.Tensor,
        filters: int,
        kernel_size: Tuple[int, int],
        dilation_rate: int,
        name: str
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """構建編碼器塊"""
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            dilation_rate=dilation_rate,
            name=f'{name}_conv'
        )(inputs)
        x = BatchNormalization(name=f'{name}_bn')(x)
        x = Activation('relu', name=f'{name}_relu')(x)
        
        # 使用same padding確保輸出大小是輸入的一半
        pool = MaxPooling2D(pool_size=(2, 2), padding='same', name=f'{name}_pool')(x)
        
        return pool, x
    
    def _build_decoder_block(
        self,
        inputs: tf.Tensor,
        skip_connection: Optional[tf.Tensor],
        filters: int,
        kernel_size: Tuple[int, int],
        name: str
    ) -> tf.Tensor:
        """構建解碼器塊"""
        # 使用same padding進行上採樣
        x = UpSampling2D(size=(2, 2), name=f'{name}_upsample')(inputs)
        
        # 確保上採樣後的特徵圖與跳躍連接的大小匹配
        if skip_connection is not None:
            target_shape = tf.shape(skip_connection)[1:3]
            x = tf.image.resize(x, target_shape, method='nearest')
            x = Concatenate(name=f'{name}_concat')([x, skip_connection])
        
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            name=f'{name}_conv'
        )(x)
        x = BatchNormalization(name=f'{name}_bn')(x)
        x = Activation('relu', name=f'{name}_relu')(x)
        
        return x
    
    def _build_encoder(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """構建編碼器"""
        skip_connections = []
        x = inputs
        
        # 編碼器層
        for i, (f, k, d) in enumerate(zip(self.filters, self.kernel_sizes, self.dilation_rates)):
            x, skip = self._build_encoder_block(
                x, f, k, d,
                name=f'encoder_{i}'
            )
            skip_connections.append(skip)
        
        # 計算最後一層的特徵圖大小
        # 假設每層下採樣2倍，計算最終特徵圖大小
        h = self.input_shape[0] // (2 ** len(self.filters))
        w = self.input_shape[1] // (2 ** len(self.filters))
        c = self.filters[-1]
        shape_before_flatten = (h, w, c)
        
        # 壓縮到潛在空間
        x = Flatten(name='flatten')(x)
        x = Dense(self.latent_dim, name='latent_dense')(x)
        
        return x, skip_connections, shape_before_flatten
    
    def _build_decoder(
        self,
        latent: tf.Tensor,
        skip_connections: List[tf.Tensor],
        shape_before_flatten: Tuple[int, int, int]
    ) -> tf.Tensor:
        """構建解碼器"""
        # 從潛在空間恢復
        h, w, c = shape_before_flatten
        units = h * w * c
        x = Dense(units, name='decoder_dense')(latent)
        
        # 重塑張量
        x = Reshape(shape_before_flatten, name='decoder_reshape')(x)
        
        # 解碼器層
        for i, (f, k, skip) in enumerate(zip(
            reversed(self.filters),
            reversed(self.kernel_sizes),
            reversed(skip_connections) if self.use_skip_connections else [None] * len(self.filters)
        )):
            x = self._build_decoder_block(
                x,
                skip if self.use_skip_connections else None,
                f, k,
                name=f'decoder_{i}'
            )
        
        # 最終輸出層
        outputs = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            padding='same',
            activation='sigmoid',
            name='decoder_output'
        )(x)
        
        return outputs
    
    def build(self) -> None:
        """構建完整的自編碼器模型"""
        inputs = Input(shape=self.input_shape, name='encoder_input')
        
        # 編碼器
        latent, skip_connections, shape_before_flatten = self._build_encoder(inputs)
        
        # 解碼器
        reconstructed = self._build_decoder(latent, skip_connections, shape_before_flatten)
        
        # 創建模型
        self.model = Model(inputs=inputs, outputs=reconstructed, name='autoencoder')
        self.encoder = Model(inputs=inputs, outputs=latent, name='encoder')
        
        # 編譯模型
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
    
    def train(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        callbacks: List = None
    ) -> tf.keras.callbacks.History:
        """訓練模型"""
        return self.model.fit(
            x_train, x_train,  # 自編碼器的輸入和輸出相同
            validation_data=(x_val, x_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """將輸入數據編碼為潛在表示"""
        return self.encoder.predict(x)
    
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """重建輸入數據"""
        return self.model.predict(x) 