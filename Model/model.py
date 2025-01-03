"""
神經網路模型定義：實現吞嚥聲分類的CNN模型

SwallowModel類特點：
1. 5層卷積層架構，使用dilated convolution
2. 每層後接BatchNorm和MaxPooling
3. 分類器使用GlobalAveragePooling + Dropout + Dense
4. 10分類輸出(正常/病患 x 5種動作)

參考自load_model1130.py的實現：
- 使用dilated convolution增加感受野
- 採用BatchNormalization提升訓練穩定性
- 使用Dropout防止過擬合
- 通過Adam優化器進行訓練
"""

from dataclasses import dataclass
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from .config import ModelConfig

@dataclass
class SwallowModel:
    config: ModelConfig
    
    def _build_conv_block(self, x, filters, kernel_size, dilation_rate):
        """构建单个卷积块"""
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
    
    def _build_conv_layers(self, inputs):
        """构建所有卷积层"""
        x = inputs
        
        # 构建主要的卷积块
        for i in range(len(self.config.filters) - 1):  # -1 是因为最后一个卷积层单独处理
            x = self._build_conv_block(
                x,
                filters=self.config.filters[i],
                kernel_size=self.config.kernel_sizes[i],
                dilation_rate=self.config.dilation_rates[i]
            )
        
        # 添加最后一个复杂性卷积层
        x = self._build_conv_block(
            x,
            filters=self.config.filters[-1],  # 256
            kernel_size=self.config.kernel_sizes[-1],
            dilation_rate=self.config.dilation_rates[-1]
        )
        
        return x
    
    def _build_classifier(self, x):
        """构建分类器部分"""
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.num_classes, activation='softmax')(x)
        return outputs
    
    def _compile_model(self, model):
        """编译模型"""
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auroc')]
        )
        
    def build(self) -> Model:
        """构建完整的模型"""
        # 创建输入层
        inputs = Input(shape=self.config.input_shape, name="input")
        
        # 构建卷积层部分
        x = self._build_conv_layers(inputs)
        
        # 构建分类器部分
        outputs = self._build_classifier(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs, name="SwallowNet")
        
        # 编译模型
        self._compile_model(model)
        
        return model

    def summary(self, model):
        """打印模型结构摘要"""
        model.summary()
    
    def predict_file(self, model, wav_file, audio_config):
        """文件級預測的三種策略"""
        # 獲取所有段的預測
        segments_pred = model.predict(wav_file)
        
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
