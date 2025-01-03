"""
基礎配置類：定義所有配置的基本結構
包含：
1. AudioConfig: 音頻處理相關配置
2. ModelConfig: 模型結構相關配置
3. TrainingConfig: 訓練過程相關配置
4. DatasetConfig: 數據集相關配置
"""

import os
import yaml
import re
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from Model.base.abstract import BaseConfig
import logging

@dataclass
class AudioConfig(BaseConfig):
    sr: int = 16000
    frame_size: int = 256
    overlap: int = 128
    fft_size: int = 256
    seq_num: int = 129
    batch_proc_size: int = 10

    def validate(self) -> None:
        if self.frame_size <= 0 or self.overlap >= self.frame_size:
            raise ValueError("Invalid frame_size or overlap")
        if self.fft_size < self.frame_size:
            raise ValueError("fft_size must be >= frame_size")

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def window_size(self) -> int:
        return self.frame_size

    @property
    def hop_length(self) -> int:
        return self.frame_size - self.overlap

@dataclass
class ModelConfig(BaseConfig):
    """模型配置類"""
    num_classes: int
    input_shape: Tuple[int, int, int]
    
    # 基礎參數
    dropout_rate: float = 0.5
    learning_rate: float = 0.0001
    feature_scale: int = 4
    save_dir: str = '/content/drive/MyDrive/MicforDysphagia/saved_models'
    
    # 網絡結構參數
    filters: Optional[List[int]] = None
    kernel_sizes: Optional[List[Tuple[int, int]]] = None
    dilation_rates: Optional[List[int]] = None
    
    # Triplet Loss 特定參數
    embedding_dim: Optional[int] = None
    margin: Optional[float] = None
    mining_strategy: Optional[str] = None
    
    # 自編碼器特定參數
    latent_dim: Optional[int] = None
    reconstruction_weight: float = 0.5
    classification_weight: float = 0.5
    use_skip_connections: bool = True
    
    def __post_init__(self):
        """後初始化處理"""
        # 確保kernel_sizes是元組列表
        if self.kernel_sizes is not None:
            self.kernel_sizes = [
                tuple(size) if isinstance(size, list) else size
                for size in self.kernel_sizes
            ]
    
    def validate(self) -> None:
        """驗證配置參數"""
        # 基礎參數驗證
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError("dropout_rate必須在0和1之間")
        if self.feature_scale <= 0:
            raise ValueError("feature_scale必須大於0")
            
        # 網絡結構驗證
        if self.filters is not None:
            if len(self.filters) == 0:
                raise ValueError("filters不能為空列表")
            if not all(f > 0 for f in self.filters):
                raise ValueError("所有的 filters 值必須大於0")
                
        if self.kernel_sizes is not None:
            if len(self.kernel_sizes) == 0:
                raise ValueError("kernel_sizes不能為空列表")
            if not all(all(k > 0 for k in ks) for ks in self.kernel_sizes):
                raise ValueError("所有的 kernel_size 值必須大於0")
                
        if self.dilation_rates is not None:
            if len(self.dilation_rates) == 0:
                raise ValueError("dilation_rates不能為空列表")
            if not all(d > 0 for d in self.dilation_rates):
                raise ValueError("所有的 dilation_rate 值必須大於0")
        
        # Triplet Loss 參數驗證
        if self.embedding_dim is not None:
            if self.embedding_dim <= 0:
                raise ValueError("embedding_dim必須大於0")
            if self.margin is None:
                raise ValueError("使用triplet loss時必須指定margin")
            if self.margin <= 0:
                raise ValueError("margin必須大於0")
            if self.mining_strategy not in [None, "hard", "semi-hard", "all"]:
                raise ValueError("mining_strategy必須是 'hard', 'semi-hard' 或 'all'")
        
        # 自編碼器參數驗證
        if self.latent_dim is not None:
            if self.latent_dim <= 0:
                raise ValueError("latent_dim必須大於0")
            if not (0 <= self.reconstruction_weight <= 1):
                raise ValueError("reconstruction_weight必須在0和1之間")
            if not (0 <= self.classification_weight <= 1):
                raise ValueError("classification_weight必須在0和1之間")
    
    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return asdict(self)

@dataclass
class TrainingConfig(BaseConfig):
    """訓練配置"""
    epochs: int = 4
    batch_size: int = 32
    early_stopping_patience: int = 3
    outside_test_ratio: float = 0.2
    validation_ratio: float = 0.2
    model_save_path: str = "trained_model.keras"
    pretrain_epochs: int = 2
    min_samples_per_class: int = 3

    def validate(self) -> None:
        if not (0 < self.validation_ratio < 1):
            raise ValueError("validation_ratio必須在0和1之間")
        if not (0 < self.outside_test_ratio < 1):
            raise ValueError("outside_test_ratio必須在0和1之間")
        if self.min_samples_per_class < 2:
            raise ValueError("min_samples_per_class必須大於等於2")

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DatasetConfig(BaseConfig):
    base_paths: List[str]
    selection_types: Dict[str, List[str]] = field(default_factory=lambda: {
        'NoMovement': ['無動作', '無吞嚥'],
        'DrySwallow': ['乾吞嚥1口', '乾吞嚥2口', '乾吞嚥3口', '乾吞嚥'],
        'WaterDrinking': ['吞水10ml', '吞水20ml', '喝水', '吞水'],
        'Cracker': ['餅乾1塊', '餅乾2塊', '餅乾'],
        'Jelly': ['吞果凍', '果凍']
    })
    action_types: List[str] = field(default_factory=lambda: [
        'NoMovement', 'DrySwallow'  # 當前只處理這兩種動作
    ])
    labels_str: List[str] = field(default_factory=lambda: [
        'Normal-No Movement', 'Normal-Dry Swallow',
        'Patient-No Movement', 'Patient-Dry Swallow'
    ])
    score_threshold: int = 3

    def get_action_type(self, selection: str) -> Optional[str]:
        """根據selection值確定動作類型"""
        if not selection:
            return None
            
        # 檢查動作類型
        for action_type, keywords in self.selection_types.items():
            if any(keyword in selection for keyword in keywords):
                return action_type
                
        return None

    def validate(self) -> None:
        # 檢查標籤數量（現在是4個標籤，2個動作 * 2個組別）
        if len(self.labels_str) != len(self.action_types) * 2:
            raise ValueError(f"labels_str必須包含{len(self.action_types) * 2}個標籤（{len(self.action_types)}個動作 * 2個組別）")
        
        # 檢查動作類型數量（現在是2種動作）
        if len(self.action_types) != 2:
            raise ValueError("action_types必須包含2個動作類型")
        
        # 檢查每個動作類型是否都有對應的關鍵詞
        for action in self.action_types:
            if action not in self.selection_types:
                raise ValueError(f"動作類型 {action} 沒有對應的關鍵詞列表")
            if not self.selection_types[action]:
                raise ValueError(f"動作類型 {action} 的關鍵詞列表為空")

    def to_dict(self) -> Dict:
        return asdict(self)

    def __post_init__(self):
        self.label_mapping = {
            (False, action): i for i, action in enumerate(self.action_types)
        }
        self.label_mapping.update({
            (True, action): i + len(self.action_types) for i, action in enumerate(self.action_types)
        })
        self.validate()

@dataclass
class Config:
    """整合所有配置的類"""
    audio: AudioConfig
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

def load_config(experiment_type: str = 'baseline') -> Config:
    """加載配置文件
    
    Args:
        experiment_type: 實驗類型，可以是 'baseline', 'autoencoder', 'triplet_loss' 等
        
    Returns:
        Config: 包含所有配置的對象
    """
    # 獲取當前文件的目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 獲取項目根目錄
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 檢查是否在Google Colab環境中
    def check_ipython():
        try:
            return 'google.colab' in str(__import__('IPython').get_ipython())
        except:
            return False
    
    is_colab = any([
        'COLAB_GPU' in os.environ,
        'COLAB_TPU_ADDR' in os.environ,
        '/content/drive' in project_root,
        check_ipython()
    ])
    
    # 構建配置文件路徑
    if is_colab:
        config_name = 'config_colab.yaml'
        logging.info("檢測到Google Colab環境，使用config_colab.yaml")
    else:
        config_name = 'config.yaml'
        logging.info("使用本地環境配置config.yaml")
    
    config_path = os.path.join(project_root, 'Model', 'experiments', experiment_type, config_name)
    
    print(f"嘗試加載配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # 創建配置對象
        audio_config = AudioConfig(**config_dict['audio'])
        
        # 根據實驗類型設置默認值
        if experiment_type == 'triplet_loss':
            model_config = ModelConfig(
                num_classes=config_dict['model']['num_classes'],
                input_shape=(config_dict['audio']['seq_num'], config_dict['audio']['seq_num'], 1),
                **{k: v for k, v in config_dict['model'].items() if k != 'num_classes'}
            )
        else:
            model_config = ModelConfig(**config_dict['model'])
            
        training_config = TrainingConfig(**config_dict['training'])
        dataset_config = DatasetConfig(**config_dict['dataset'])
        
        return Config(
            audio=audio_config,
            model=model_config,
            training=training_config,
            dataset=dataset_config
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    except Exception as e:
        raise Exception(f"加載配置文件時出錯: {str(e)}") 