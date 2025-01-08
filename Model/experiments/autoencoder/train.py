"""
訓練自動編碼器模型的主腳本
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import random
import os
import sys

import tensorflow as tf
import numpy as np

from Model.data_loaders.autoencoder_loader import AutoEncoderDataLoader
from base.class_config import get_num_classes

def setup_logger() -> logging.Logger:
    """設置日誌記錄器

    Returns:
        logging.Logger: 配置好的日誌記錄器
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 創建控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 創建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 添加處理器到日誌記錄器
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger

def setup_save_dir() -> str:
    """設置保存目錄

    Returns:
        str: 保存目錄的路徑
    """
    save_dir = os.path.join('saved_models', 'autoencoder')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_history(history, save_dir: str):
    """保存訓練歷史

    Args:
        history: 訓練歷史對象
        save_dir (str): 保存目錄
    """
    history_file = os.path.join(save_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history.history, f, indent=2)

# 此代碼實現了自動編碼器的訓練流程，包含特徵維度統一化和評估功能

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import random

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
from Model.utils.evaluation import evaluate_model
from base.class_config import get_num_classes

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

def split_subjects(
    patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """根據受試者ID分割數據
    
    Args:
        patient_ids: 受試者ID列表
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        random_seed: 隨機種子
        
    Returns:
        (train_subjects, val_subjects, test_subjects): 分割後的受試者ID列表
    """
    # 獲取唯一的受試者ID
    unique_subjects = list(set(patient_ids))
    
    # 確保至少有3個受試者
    if len(unique_subjects) < 3:
        raise ValueError(f"受試者數量（{len(unique_subjects)}）不足，無法進行分割。至少需要3個受試者。")
    
    # 設置隨機種子
    np.random.seed(random_seed)
    np.random.shuffle(unique_subjects)
    
    # 計算分割點，確保每個子集至少有一個受試者
    n_subjects = len(unique_subjects)
    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))
    
    # 調整分割以確保測試集至少有一個受試者
    if n_train + n_val >= n_subjects:
        if n_subjects >= 3:
            n_train = n_subjects - 2
            n_val = 1
        else:
            raise ValueError(f"受試者數量（{n_subjects}）不足，無法保證每個子集至少有一個受試者")
    
    # 分割受試者
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train+n_val]
    test_subjects = unique_subjects[n_train+n_val:]
    
    return train_subjects, val_subjects, test_subjects

def prepare_data(
    features: np.ndarray,
    labels: np.ndarray,
    filenames: List[str],
    patient_ids: List[str],
    logger: Optional[logging.Logger] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """準備訓練、驗證和測試數據

    Args:
        features (np.ndarray): 特徵數據
        labels (np.ndarray): 標籤數據
        filenames (List[str]): 文件名列表
        patient_ids (List[str]): 受試者ID列表
        logger (Optional[logging.Logger], optional): 日誌記錄器. Defaults to None.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            訓練數據、驗證數據和測試數據的元組，每個元組包含特徵和標籤
    """
    # 檢查數據有效性
    if len(features) == 0:
        raise ValueError("沒有有效的特徵數據")
    if len(labels) == 0:
        raise ValueError("沒有有效的標籤數據")
    if len(filenames) == 0:
        raise ValueError("沒有有效的文件名")
    if len(patient_ids) == 0:
        raise ValueError("沒有有效的受試者ID")
    
    # 獲取唯一的受試者ID
    unique_patients = list(set(patient_ids))
    if len(unique_patients) < 3:
        raise ValueError("受試者數量不足，無法進行訓練/驗證/測試集分割")
    
    # 隨機打亂受試者順序
    random.shuffle(unique_patients)
    
    # 分割受試者
    train_size = int(len(unique_patients) * 0.7)
    val_size = int(len(unique_patients) * 0.15)
    
    train_patients = unique_patients[:train_size]
    val_patients = unique_patients[train_size:train_size + val_size]
    test_patients = unique_patients[train_size + val_size:]
    
    if logger:
        logger.info(f"訓練集受試者: {len(train_patients)} 人")
        logger.info(f"驗證集受試者: {len(val_patients)} 人")
        logger.info(f"測試集受試者: {len(test_patients)} 人")
    
    # 根據受試者ID分割數據
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
    
    # 檢查索引有效性
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("數據分割後某個集合為空")
    
    # 創建數據集
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    if logger:
        logger.info(f"訓練集樣本數: {len(train_features)}")
        logger.info(f"驗證集樣本數: {len(val_features)}")
        logger.info(f"測試集樣本數: {len(test_features)}")
    
    return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)

def setup_callbacks(save_dir: str) -> List[tf.keras.callbacks.Callback]:
    """設置訓練回調函數

    Args:
        save_dir (str): 保存目錄

    Returns:
        List[tf.keras.callbacks.Callback]: 回調函數列表
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'logs'),
            histogram_freq=1
        )
    ]
    return callbacks

def create_model(input_shape: Tuple[int, ...]) -> tf.keras.Model:
    """創建模型

    Args:
        input_shape (Tuple[int, ...]): 輸入形狀

    Returns:
        tf.keras.Model: 創建的模型
    """
    num_classes = get_num_classes()
    
    model = tf.keras.Sequential([
        # 輸入層
        tf.keras.layers.Input(shape=input_shape),
        
        # 特徵提取層
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        
        # 全局池化層
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # 分類層
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_model(model: tf.keras.Model, test_data: Dict, save_dir: str):
    """評估模型

    Args:
        model (tf.keras.Model): 訓練好的模型
        test_data (Dict): 測試數據
        save_dir (str): 保存目錄
    """
    # 評估模型
    test_loss, test_accuracy = model.evaluate(
        test_data['encoder_input'],
        test_data['classifier_output']
    )
    
    # 保存評估結果
    evaluation_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy)
    }
    
    evaluation_file = os.path.join(save_dir, 'evaluation_results.json')
    with open(evaluation_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

def train_model(
    model: AutoencoderModel,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    config: dict,
    logger: logging.Logger
) -> Tuple[Dict, Dict]:
    """訓練並評估模型
    
    Args:
        model: 模型實例
        train_data: 訓練數據
        val_data: 驗證數據
        config: 配置字典
        logger: 日誌記錄器
        
    Returns:
        Tuple[Dict, Dict]: (訓練歷史, 評估結果)
    """
    # 設置保存目錄
    save_dir = Path(config.get('save_dir', 'saved_models'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置回調
    callbacks = setup_callbacks(str(save_dir))
    
    # 訓練模型
    history = model.fit(
        x=train_data['encoder_input'],
        y=train_data['classifier_output'],
        validation_data=(
            val_data['encoder_input'],
            val_data['classifier_output']
        ),
        batch_size=config.get('batch_size', 32),
        epochs=2,  # 簡化為2個epoch
        callbacks=callbacks,
        verbose=1
    )
    
    # 評估模型
    evaluation_results = evaluate_model(
        model,
        val_data,
        val_data['classifier_output'],
        str(save_dir),
        logger
    )
    
    return history.history, evaluation_results

def load_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """加載數據

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]: 特徵、標籤、文件名和受試者ID
    """
    data_dir = os.path.join('WavData', 'AutoEncoderData')
    original_data_dir = os.path.join('WavData', 'OriginalData')
    
    data_loader = AutoEncoderDataLoader(data_dir=data_dir, original_data_dir=original_data_dir)
    features, labels, filenames, patient_ids = data_loader.load_data()
    
    if len(features) == 0:
        raise ValueError("沒有找到有效的數據")
    
    return features, labels, filenames, patient_ids

def main():
    """主函數"""
    # 設置日誌記錄器
    logger = setup_logger()
    
    try:
        # 加載數據
        features, labels, filenames, patient_ids = load_data()
        logger.info(f"加載了 {len(features)} 個樣本")
        logger.info(f"特徵形狀: {features.shape}")
        logger.info(f"標籤形狀: {labels.shape}")
        logger.info(f"文件名數量: {len(filenames)}")
        logger.info(f"受試者數量: {len(set(patient_ids))}")

        # 標準化特徵維度
        logger.info(f"原始特徵形狀: {features.shape}")
        norm_features = normalize_feature_dim(features, target_dim=740)
        logger.info(f"處理後的特徵形狀: {norm_features.shape}")
        
        # 創建全局特徵
        global_features = np.mean(norm_features, axis=1, keepdims=True)
        expanded_global = np.repeat(global_features, norm_features.shape[1], axis=1)
        
        # 合併局部和全局特徵
        combined_features = np.concatenate([norm_features, expanded_global], axis=-1)
        
        # 準備數據集
        (train_features, train_labels), (val_features, val_labels), (test_features, test_labels) = prepare_data(
            combined_features,
            labels,
            filenames,
            patient_ids,
            logger=logger
        )

        # 轉換標籤為 one-hot 編碼
        num_classes = get_num_classes()
        train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
        val_labels_onehot = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)
        test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

        # 置保存目錄
        save_dir = setup_save_dir()
        
        # 創建和編譯模型
        model = create_model(input_shape=train_features.shape[1:])
        
        # 設置回調函數
        callbacks = setup_callbacks(save_dir)
        
        # 訓練模型
        history = model.fit(
            x=train_features,
            y=train_labels_onehot,
            validation_data=(val_features, val_labels_onehot),
            epochs=2,
            batch_size=32,
            callbacks=callbacks
        )
        
        # 評估模型
        test_loss, test_accuracy = model.evaluate(test_features, test_labels_onehot)
        logger.info(f"測試集損失: {test_loss:.4f}")
        logger.info(f"測試集準確率: {test_accuracy:.4f}")
        
        # 保存評估結果
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }
        evaluation_file = os.path.join(save_dir, 'evaluation_results.json')
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # 保存訓練歷史
        save_history(history, save_dir)
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        logger.error(f"Traceback:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 