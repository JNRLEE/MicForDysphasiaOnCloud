"""
配置文件：定義系統所需的所有配置類
包含：
1. AudioConfig: 音訊處理參數配置
2. ModelConfig: 神經網路結構配置 
3. TrainingConfig: 訓練相關參數配置
4. DatasetConfig: 數據集相關配置
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

@dataclass
class AudioConfig:
    """音频处理的相关配置"""
    sr: int = 16000                # 采样率
    frame_size: int = 256          # 帧大小
    overlap: int = 128             # 重叠大小
    fft_size: int = 256           # FFT大小
    seq_num: int = 129            # 序列长度 (overlap + 1)
    batch_proc_size: int = 10     # 批处理大小
    
    @property
    def window_size(self) -> int:
        return self.frame_size
    
    @property
    def hop_length(self) -> int:
        return self.frame_size - self.overlap

@dataclass
class ModelConfig:
    """模型结构的相关配置"""
    input_shape: Tuple[Optional[int], int, int, int]
    num_classes: int = 10
    filters: List[int] = field(default_factory=lambda: [32, 64, 96, 128])
    feature_scale: int = 4
    dropout_rate: float = 0.5
    learning_rate: float = 0.0001
    kernel_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (5, 5), (3, 3), (3, 3), (3, 3), (3, 3)
    ])
    dilation_rates: List[int] = field(default_factory=lambda: [2, 2, 4, 4, 1])
    
    def __post_init__(self):
        """初始化后的处理"""
        self.filters = [int(x / self.feature_scale) for x in self.filters]
        # 添加最后一个卷积层的filter数
        self.filters.append(256)
        
        # 验证配置
        if len(self.kernel_sizes) != len(self.dilation_rates):
            raise ValueError("kernel_sizes和dilation_rates长度必须相同")

@dataclass
class TrainingConfig:
    """训练相关的配置"""
    epochs: int = 4
    batch_size: int = 32
    early_stopping_patience: int = 3
    outside_test_ratio: float = 0.2
    validation_ratio: float = 0.2
    model_save_path: str = '/content/drive/MyDrive/MicforDysphagia/trained_model.keras'
    
    def __post_init__(self):
        """验证配置参数"""
        if not (0 < self.validation_ratio < 1):
            raise ValueError("validation_ratio必须在0和1之间")
        if not (0 < self.outside_test_ratio < 1):
            raise ValueError("outside_test_ratio必须在0和1之间")

@dataclass
class DatasetConfig:
    """数据集相关的配置"""
    base_paths: List[str]
    selection_types: Dict[str, List[str]] = field(default_factory=lambda: {
        'NoMovement': ['無動作', '無吞嚥'],
        'DrySwallow': ['乾吞嚥1口', '乾吞嚥2口', '乾吞嚥3口', '乾吞嚥'],
        'Cracker': ['餅乾1塊', '餅乾2塊', '餅乾'],
        'Jelly': ['吞果凍', '果凍'],
        'WaterDrinking': ['吞水10ml', '吞水20ml', '喝水', '吞水']
    })
    action_types: List[str] = field(default_factory=lambda: [
        'NoMovement', 'DrySwallow', 'Cracker', 'Jelly', 'WaterDrinking'
    ])
    labels_str: List[str] = field(default_factory=lambda: [
        'Normal-No Movement', 'Normal-Dry Swallow', 'Normal-Cracker',
        'Normal-Jelly', 'Normal-Water Drinking', 'Patient-No Movement',
        'Patient-Dry Swallow', 'Patient-Cracker', 'Patient-Jelly',
        'Patient-Water Drinking'
    ])
    score_threshold: int = 3  # 区分正常和病人的分数阈值

    def __post_init__(self):
        """初始化标签映射"""
        self.label_mapping = {
            (False, action): i for i, action in enumerate(self.action_types)
        }
        self.label_mapping.update({
            (True, action): i + 5 for i, action in enumerate(self.action_types)
        })
        
        # 验证配置
        if len(self.labels_str) != 10:
            raise ValueError("labels_str必须包含10个标签")
        if len(self.action_types) != 5:
            raise ValueError("action_types必须包含5个动作类型")
