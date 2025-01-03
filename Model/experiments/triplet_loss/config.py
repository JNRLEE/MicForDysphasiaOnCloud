"""
三元組損失模型配置：定義模型的超參數和配置
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TripletLossConfig:
    """三元組損失模型配置類"""
    
    # 輸入配置
    input_height: int = 128
    input_width: int = 128
    input_channels: int = 1
    
    # 模型配置
    filters: List[int] = (32, 64, 128, 256, 512)
    kernel_sizes: List[Tuple[int, int]] = ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3))
    dilation_rates: List[Tuple[int, int]] = ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))
    dropout_rate: float = 0.3
    embedding_size: int = 128
    
    # 訓練配置
    learning_rate: float = 0.001
    margin: float = 0.5
    
    # 保存路徑
    save_dir: str = '/content/drive/MyDrive/MicforDysphagia/saved_models' 