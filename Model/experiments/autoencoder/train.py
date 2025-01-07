# 此代碼實現了自動編碼器的訓練流程，包含特徵維度統一化

import os
import logging
from pathlib import Path
from typing import Tuple, Dict
import json

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.decomposition import PCA

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.autoencoder.config import load_config
from experiments.autoencoder.model import AutoencoderModel
from data_loaders.autoencoder_loader import AutoEncoderDataLoader

def setup_logging():
    """設置日誌配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def normalize_feature_dim(features: np.ndarray, target_dim: int = 370) -> np.ndarray:
    """標準化特徵維度
    
    Args:
        features: 原始特徵 [batch_size, time_steps, feature_dim]
        target_dim: 目標特徵維度
    Returns:
        normalized_features: 標準化後的特徵
    """
    if features is None:
        return None
        
    original_shape = features.shape
    current_dim = original_shape[-1]
    
    if current_dim > target_dim:
        # 使用PCA降維
        data_2d = features.reshape(-1, current_dim)
        pca = PCA(n_components=target_dim)
        data_reduced = pca.fit_transform(data_2d)
        return data_reduced.reshape(original_shape[0], original_shape[1], target_dim)
    
    elif current_dim < target_dim:
        # 使用零填充
        padded = np.zeros((original_shape[0], original_shape[1], target_dim))
        padded[..., :current_dim] = features
        return padded
        
    return features

def prepare_data(
    features: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.2,
    logger: logging.Logger = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """準備訓練和驗證數據"""
    if logger:
        logger.info(f"原始特徵形狀: {features.shape}")
    
    # 標準化特徵維度
    norm_features = normalize_feature_dim(features)
    
    # 計算分割索引
    split_idx = int(len(norm_features) * (1 - val_split))
    
    # 準備訓練數據
    train_data = {
        'encoder_input': norm_features[:split_idx],
        'classifier_output': tf.keras.utils.to_categorical(labels[:split_idx])
    }
    
    # 準備驗證數據
    val_data = {
        'encoder_input': norm_features[split_idx:],
        'classifier_output': tf.keras.utils.to_categorical(labels[split_idx:])
    }
    
    if logger:
        logger.info(f"訓練數據形狀:")
        for key, value in train_data.items():
            logger.info(f"  {key}: {value.shape}")
    
    return train_data, val_data

def setup_callbacks(save_dir: str) -> list:
    """設置訓練回調函數"""
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'model_{epoch:02d}_{val_loss:.4f}.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    return callbacks

def train_model(
    model: AutoencoderModel,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    config: dict
) -> None:
    """訓練模型"""
    callbacks = setup_callbacks(config.get('save_dir', 'saved_models'))
    
    # 設定簡化的訓練參數
    batch_size = config.get('batch_size', 32)
    epochs = 2  # 簡化為2個epoch
    
    # 訓練模型
    history = model.fit(
        x=train_data['encoder_input'],
        y=train_data['classifier_output'],
        validation_data=(
            val_data['encoder_input'],
            val_data['classifier_output']
        ),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def main():
    """主函數"""
    logger = setup_logging()
    logger.info("開始訓練自動編碼器模型")
    
    try:
        # 加載配置
        config = load_config()
        logger.info("成功加載配置")
        
        # 加載數據
        data_loader = AutoEncoderDataLoader(
            config.data_dir,
            config.original_data_dir
        )
        features, labels, filenames = data_loader.load_data()  # 修改為接收3個返回值
        logger.info(f"加載了 {len(features)} 個樣本")
        
        # 更新模型配置中的輸入形狀
        model_config = config.get('model', {})
        model_config['input_shape'] = (512, 370)  # 使用原始特徵維度
        
        # 準備訓練和驗證數據
        train_data, val_data = prepare_data(
            features,
            labels,
            config.get('training', {}).get('validation_ratio', 0.2),
            logger
        )
        
        # 構建模型
        model = AutoencoderModel(model_config)
        model.build()
        logger.info("成功構建模型")
        
        # 訓練模型
        history = train_model(model, train_data, val_data, model_config)
        logger.info("模型訓練完成")
        
        # 保存訓練歷史
        history_file = os.path.join(model_config.get('save_dir', 'saved_models'), 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history.history, f)
        logger.info(f"訓練歷史已保存到 {history_file}")
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main() 