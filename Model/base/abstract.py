"""
抽象基類：定義模型和配置的基本接口
"""

from abc import ABC, abstractmethod
from typing import Dict

class BaseConfig(ABC):
    """配置基類"""
    
    @abstractmethod
    def validate(self) -> None:
        """驗證配置參數的有效性"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict:
        """將配置轉換為字典格式"""
        pass

class BaseModel(ABC):
    """模型基類"""
    
    @abstractmethod
    def build(self):
        """構建模型"""
        pass
    
    @abstractmethod
    def train(self, train_data, val_data, **kwargs):
        """訓練模型"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data):
        """評估模型"""
        pass 