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
from tensorflow.keras import backend as K
from typing import List, Dict, Optional, Tuple

def build_model(config: Dict) -> Tuple[Model, Model, Model]:
    """構建自編碼器模型
    
    Args:
        config: 配置字典，包含模型參數
        
    Returns:
        Tuple[Model, Model, Model]: (自編碼器, 編碼器, 解碼器)
    """
    autoencoder = AutoencoderModel(config.get('model', {}))
    autoencoder.build()
    return autoencoder.model, autoencoder.encoder, None  # 目前不需要單獨的解碼器

class AutoencoderModel:
    def __init__(self, config: Dict):
        """初始化自編碼器模型
        
        Args:
            config: 配置字典，包含模型參數
        """
        self.input_shape = config.get('input_shape', 674816)  # 默認輸入形狀
        self.latent_dim = config.get('latent_dim', 128)  # 潛在空間維度
        self.filters = config.get('filters', [64, 32, 16])  # 卷積層過濾器數量
        self.kernel_sizes = config.get('kernel_sizes', [(3, 3), (3, 3), (3, 3)])  # 卷積核大小
        self.dilation_rates = config.get('dilation_rates', [1, 2, 4])  # 擴張率
        self.learning_rate = config.get('learning_rate', 1e-4)  # 學習率
        
        # 確保模型參數列表長度一致
        assert len(self.filters) == len(self.kernel_sizes) == len(self.dilation_rates), \
            "filters、kernel_sizes和dilation_rates的長度必須相同"
        
        # 初始化模型
        self.model = None
        self.encoder = None
        self.decoder = None
    
    def _validate_config(self):
        """驗證配置參數的完整性和有效性"""
        # 檢查必要參數
        required_params = {
            'latent_dim': 128,
            'filters': [32, 64, 96, 128],
            'kernel_sizes': [[3, 3]] * 4,
            'dilation_rates': [1, 2, 2, 4]
        }
        
        # 為缺失的參數設置默認值
        for param, default_value in required_params.items():
            if param not in self.config:
                self.config[param] = default_value
                
        # 檢查參數的一致性
        if len(self.config['filters']) != len(self.config['kernel_sizes']):
            raise ValueError(
                f"filters ({len(self.config['filters'])}) 和 "
                f"kernel_sizes ({len(self.config['kernel_sizes'])}) 的長度必須相同"
            )
        if len(self.config['filters']) != len(self.config['dilation_rates']):
            raise ValueError(
                f"filters ({len(self.config['filters'])}) 和 "
                f"dilation_rates ({len(self.config['dilation_rates'])}) 的長度必須相同"
            )
            
        # 檢查參數的有效性
        if self.config['latent_dim'] <= 0:
            raise ValueError(f"latent_dim ({self.config['latent_dim']}) 必須大於0")
        if not all(f > 0 for f in self.config['filters']):
            raise ValueError("所有的 filters 值必須大於0")
        if not all(all(k > 0 for k in ks) for ks in self.config['kernel_sizes']):
            raise ValueError("所有的 kernel_size 值必須大於0")
        if not all(d > 0 for d in self.config['dilation_rates']):
            raise ValueError("所有的 dilation_rate 值必須大於0")
    
    def _build_encoder_block(
        self,
        x,
        filters,
        kernel_size,
        dilation_rate,
        name_prefix
    ):
        """建構編碼器塊
        
        包含卷積、批量歸一化和池化操作
        """
        # 卷積層
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='same',
            name=f'{name_prefix}_conv1'
        )(x)
        x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        # 池化層
        x = MaxPooling1D(pool_size=2, name=f'{name_prefix}_pool1')(x)
        
        return x
    
    def _build_decoder_block(
        self,
        x,
        filters,
        kernel_size,
        dilation_rate,
        name_prefix
    ):
        """建構解碼器塊
        
        包含上採樣、卷積和批量歸一化操作
        """
        # 上採樣
        x = UpSampling1D(size=2, name=f'{name_prefix}_upsample')(x)
        
        # 卷積層
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='same',
            name=f'{name_prefix}_conv1'
        )(x)
        x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        return x
    
    def _build_encoder(self, inputs):
        """建構編碼器
        
        將輸入轉換為潛在空間的表示
        """
        x = inputs
        skip_connections = []
        
        # 編碼器塊
        for i, (filters, kernel_size, dilation_rate) in enumerate(
            zip(self.filters, self.kernel_sizes, self.dilation_rates)
        ):
            x = self._build_encoder_block(
                x,
                filters,
                kernel_size,
                dilation_rate,
                f'encoder_block_{i}'
            )
            skip_connections.append(x)
        
        # 記錄展平前的形狀
        shape_before_flatten = K.int_shape(x)
        
        # 展平並降維到潛在空間
        x = Flatten()(x)
        latent = Dense(self.latent_dim, name='encoder_dense')(x)
        
        return latent, skip_connections, shape_before_flatten
    
    def _build_decoder(self, latent, skip_connections, shape_before_flatten):
        """建立解碼器部分"""
        # 從潛在空間重建
        x = Dense(shape_before_flatten[1] * shape_before_flatten[2], activation='relu')(latent)
        x = Reshape((shape_before_flatten[1], shape_before_flatten[2]))(x)

        # 反向應用卷積層
        for i, (filters, kernel_size, dilation_rate) in enumerate(zip(
            reversed(self.filters[:-1]),  # 不包括最後一層
            reversed(self.kernel_sizes[:-1]),
            reversed(self.dilation_rates[:-1])
        )):
            # 上採樣
            x = UpSampling1D(2)(x)
            
            # 確保跳躍連接的形狀匹配
            skip = skip_connections[-(i+1)]
            if K.int_shape(x)[1] != K.int_shape(skip)[1]:
                # 如果形狀不匹配，調整skip connection的大小
                target_shape = K.int_shape(x)[1]
                skip = Lambda(lambda x: x[:, :target_shape, :])(skip)
            
            # 添加跳躍連接
            x = Add(name=f'skip_connection_{i}')([x, skip])
            
            # 卷積層
            x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        # 最後一層使用單一通道的卷積來重建輸入
        decoded = Conv1D(1, self.kernel_sizes[-1], padding='same', dilation_rate=self.dilation_rates[-1])(x)
        
        return decoded
    
    def build(self):
        """建構自動編碼器模型
        
        包含編碼器和解碼器兩個主要部分
        """
        # 定義輸入層
        inputs = Input(shape=(self.input_shape,), name='encoder_input')
        
        # 重塑輸入以適應1D卷積
        reshaped_inputs = Reshape((self.input_shape, 1), name='reshape_input')(inputs)
        
        # 構建編碼器
        latent, skip_connections, shape_before_flatten = self._build_encoder(reshaped_inputs)
        
        # 構建解碼器
        decoded = self._build_decoder(latent, skip_connections, shape_before_flatten)
        
        # 創建模型
        self.model = Model(inputs=inputs, outputs=decoded, name='autoencoder')
        
        # 編譯模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # 打印模型摘要
        self.model.summary()
    
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