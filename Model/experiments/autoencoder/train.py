# 此代碼實現了自動編碼器的訓練流程

import os
import logging
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

from .config import load_config
from .model import AutoencoderModel
from ...data_loaders.autoencoder_loader import AutoEncoderDataLoader

def setup_logging():
    """設置日誌配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def setup_callbacks(save_dir: str) -> list:
    """設置訓練回調函數"""
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(save_dir, 'logs'),
            histogram_freq=1
        )
    ]
    return callbacks

def train_model(model: AutoencoderModel,
               train_data: Tuple[tf.Tensor, tf.Tensor],
               val_data: Tuple[tf.Tensor, tf.Tensor],
               config) -> None:
    """訓練模型"""
    callbacks = setup_callbacks(config.get('save_dir', 'saved_models'))
    
    model.train(
        x_train=train_data[0],
        x_val=val_data[0],
        batch_size=config.get('batch_size', 32),
        epochs=config.get('epochs', 100),
        callbacks=callbacks
    )

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
        features, labels, filenames = data_loader.load_data()
        logger.info(f"加載了 {len(features)} 個樣本")
        
        # 劃分訓練集和驗證集
        val_split = config.get('training', {}).get('validation_ratio', 0.2)
        split_idx = int(len(features) * (1 - val_split))
        
        train_data = (features[:split_idx], features[:split_idx])
        val_data = (features[split_idx:], features[split_idx:])
        
        logger.info(f"訓練集大小: {len(train_data[0])}")
        logger.info(f"驗證集大小: {len(val_data[0])}")
        
        # 構建模型
        model = AutoencoderModel(config.get('model', {}))
        model.build()
        logger.info("成功構建模型")
        
        # 訓練模型
        train_model(model, train_data, val_data, config.get('model', {}))
        logger.info("模型訓練完成")
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        raise

if __name__ == '__main__':
    main() 