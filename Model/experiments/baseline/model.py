"""
基線模型實現：使用CNN進行吞嚥聲分類
特點：
1. 5層卷積層架構，使用dilated convolution
2. 每層後接BatchNorm和MaxPooling
3. 分類器使用GlobalAveragePooling + Dropout + Dense
4. 使用categorical crossentropy作為損失函數
"""

from typing import Dict, Tuple, Any
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

from Model.base.abstract import BaseModel
from Model.base.config import ModelConfig

class BaselineModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
    
    def _build_conv_block(self, x, filters: int, kernel_size: Tuple[int, int], 
                         dilation_rate: int) -> Model:
        """構建單個卷積塊"""
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            dilation_rate=dilation_rate,
            activation='relu',
            padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        return x
    
    def _build_conv_layers(self, inputs) -> Model:
        """構建所有卷積層"""
        x = inputs
        
        # 構建主要的卷積塊
        for i in range(len(self.config.filters) - 1):
            x = self._build_conv_block(
                x,
                filters=self.config.filters[i],
                kernel_size=self.config.kernel_sizes[i],
                dilation_rate=self.config.dilation_rates[i]
            )
        
        # 添加最後一個卷積層
        x = self._build_conv_block(
            x,
            filters=self.config.filters[-1],
            kernel_size=self.config.kernel_sizes[-1],
            dilation_rate=self.config.dilation_rates[-1]
        )
        
        return x
    
    def _build_classifier(self, x) -> Model:
        """構建分類器部分"""
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.num_classes, activation='softmax')(x)
        return outputs
    
    def build(self) -> Model:
        """構建完整的模型"""
        # 創建輸入層
        inputs = Input(shape=self.config.input_shape, name="input")
        
        # 構建卷積層部分
        x = self._build_conv_layers(inputs)
        
        # 構建分類器部分
        outputs = self._build_classifier(x)
        
        # 創建模型
        self.model = Model(inputs=inputs, outputs=outputs, name="BaselineSwallowNet")
        
        # 編譯模型
        optimizer = Adam(learning_rate=self.config.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auroc')]
        )
        
        return self.model

    def train(self, train_data: Tuple, val_data: Tuple, **kwargs) -> Dict:
        """訓練模型"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 設置early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=kwargs.get('patience', 3),
            restore_best_weights=True
        )
        
        # 訓練模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=kwargs.get('epochs', 4),
            batch_size=kwargs.get('batch_size', 32),
            class_weight=kwargs.get('class_weight'),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history.history

    def evaluate(self, test_data: Tuple) -> Dict[str, float]:
        """評估模型"""
        X_test, y_test = test_data
        metrics = self.model.evaluate(X_test, y_test, verbose=1)
        return {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'auroc': metrics[2]
        }

    def predict_file(self, wav_file: str, audio_config: Any) -> Dict:
        """文件級預測"""
        # 獲取所有段的預測
        segments_pred = self.model.predict(wav_file)
        
        # 策略1：投票法
        vote_pred = np.argmax(np.bincount(np.argmax(segments_pred, axis=1)))
        
        # 策略2：平均概率
        mean_pred = np.argmax(np.mean(segments_pred, axis=0))
        
        # 策略3：加權投票（根據預測確信度加權）
        confidence_weights = np.max(segments_pred, axis=1)
        weighted_votes = np.zeros(self.config.num_classes)
        for pred, weight in zip(np.argmax(segments_pred, axis=1), confidence_weights):
            weighted_votes[pred] += weight
        weighted_pred = np.argmax(weighted_votes)
        
        return {
            'vote': vote_pred,
            'mean': mean_pred,
            'weighted': weighted_pred,
            'segments_pred': segments_pred
        } 