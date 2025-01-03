"""
主程序：訓練和評估三元組損失模型
"""

import os
import logging
import yaml
import numpy as np
from typing import Dict, Any

from Model.base.data_loader import TripletDataLoader
from Model.experiments.triplet_loss.model import TripletLossModel
from Model.experiments.triplet_loss.config import TripletLossConfig

def load_config(config_path: str) -> Dict[str, Any]:
    """加載YAML配置文件
    
    Args:
        config_path: 配置文件路徑
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主函數"""
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加載配置
    config_path = os.path.join(
        os.path.dirname(__file__),
        'config_colab.yaml'
    )
    config_dict = load_config(config_path)
    config = TripletLossConfig(**config_dict)
    
    # 創建數據加載器
    data_loader = TripletDataLoader(config)
    
    # 加載數據
    x_train = data_loader.data
    y_train = data_loader.labels
    
    # 創建模型
    model = TripletLossModel(config)
    
    # 訓練模型
    history = model.train(
        data_loader=data_loader,
        x_train=x_train,
        y_train=y_train,
        batch_size=4,
        epochs=4
    )
    
    # 保存訓練歷史
    history_path = os.path.join(config.save_dir, 'training_history.npy')
    np.save(history_path, history.history)
    logging.info(f"訓練歷史已保存至: {history_path}")

if __name__ == '__main__':
    main() 