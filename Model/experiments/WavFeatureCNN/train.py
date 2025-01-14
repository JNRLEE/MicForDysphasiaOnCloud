"""
此腳本用於訓練吞嚥聲音分類模型
主要功能：
1. 加載和預處理數據
2. 將特徵維度標準化為配置文件中指定的維度
3. 訓練模型並保存結果
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import random
import yaml
from collections import Counter
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# 取專案根目錄的絕對路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 切換到專案根目錄
os.chdir(PROJECT_ROOT)

# 添加項目根目錄到 Python 路徑
sys.path.append(PROJECT_ROOT)

from Model.data_loaders.WavFeatureCNN_loader import AutoEncoderDataLoader
from Model.base.class_config import get_num_classes, get_active_class_names, CLASS_CONFIG
from Model.base.visualization import VisualizationTool
from Model.experiments.WavFeatureCNN.model import AutoencoderModel

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
    """設置保存目錄"""
    save_dir = os.path.join('saved_models', 'WavFeatureCNN')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_history(history, save_dir: str):
    """保存訓練歷史

    Args:
        history: 訓練歷史對象
        save_dir (str): 保存目錄
    """
    # 將 float32 轉換為 float
    history_dict = {}
    for key, value in history.history.items():
        history_dict[key] = [float(v) for v in value]
    
    history_file = os.path.join(save_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history_dict, f, indent=2)

def print_class_config():
    """打印當前使用的類別配置"""
    print("\n=== 當前類別配置 ===")
    for class_name, is_active in CLASS_CONFIG.items():
        status = "啟用" if is_active == 1 else "停用"
        print(f"{class_name}: {status}")
    print("==================\n")

def print_split_info(
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: List[str]
):
    """打印數據集分割的詳細信息
    
    Args:
        split_name: 分割名稱 (train/val/test)
        features: 特徵數據
        labels: 標籤數據
        patient_ids: 受試者ID列表
    """
    print(f"\n=== {split_name} 集資訊 ===")
    print(f"樣本數量: {len(features)}")
    print(f"特徵形狀: {features.shape}")
    
    # 打印受試者ID
    unique_patients = sorted(set(patient_ids))
    print(f"受試者 ({len(unique_patients)}): {', '.join(unique_patients)}")
    
    # 統計每個類別的樣樣本數
    label_names = get_active_class_names()
    label_counts = Counter(labels)
    print("\n各類別樣本數:")
    for label_idx, count in sorted(label_counts.items()):
        print(f"{label_names[label_idx]}: {count} 筆")
    print("==================\n")

def print_confusion_matrix_text(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]):
    """以文字形式打印混淆矩陣
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        label_names: 標籤名稱列表
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=== 混淆矩陣 ===")
    # 打印標籤
    label_width = max(len(name) for name in label_names)
    print(" " * (label_width + 2), end="")
    for name in label_names:
        print(f"{name:^10}", end=" ")
    print("\n")
    
    # 打印矩陣內容
    for i, row in enumerate(cm):
        print(f"{label_names[i]:<{label_width}}", end="  ")
        for val in row:
            print(f"{val:^10}", end=" ")
        print()
    print("==================\n")

def normalize_feature_dim(features: np.ndarray, config: Dict) -> np.ndarray:
    """標準化特徵維度到配置文件指定的維度
    
    Args:
        features: 原始特徵 [batch_size, time_steps, feature_dim]
        config: 配置字典，包含目標維度設定
        
    Returns:
        normalized_features: 標準化後的特徵
    """
    if features is None:
        return None
        
    original_shape = features.shape
    current_dim = original_shape[-1]
    target_dim = config['feature_processing']['target_dim']
    
    if current_dim > target_dim:
        # 截斷較長的特徵
        return features[..., :target_dim]
    elif current_dim < target_dim:
        # 使用零填充
        padded = np.zeros((original_shape[0], original_shape[1], target_dim))
        padded[..., :current_dim] = features
        return padded
        
    return features

def sliding_window(data: np.ndarray, window_size: int = 129, stride: int = 64) -> np.ndarray:
    """使用滑動窗口分割數據
    
    Args:
        data: 輸入數據，形狀為 [samples, channels, sequence_length]
        window_size: 窗口大小，默認為 129 (對應 example.py 中的 seq_num)
        stride: 步長，默認為 64 (window_size//2，確保50%重疊)
        
    Returns:
        windows: 分割後的窗口，形狀為 [num_windows, channels, window_size]
    """
    # 確保數據維度正確
    if len(data.shape) != 3:
        raise ValueError(f"輸入數據維度應為3，當前維度為: {len(data.shape)}")
    
    samples, channels, sequence_length = data.shape
    
    # 如果序列長度小於窗口大小，進行填充
    if sequence_length < window_size:
        pad_size = window_size - sequence_length
        data = np.pad(data, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
        sequence_length = window_size
    
    # 計算窗口數量
    num_windows = (sequence_length - window_size) // stride + 1
    
    # 創建結果數組
    windows = np.zeros((samples * num_windows, channels, window_size))
    
    # 對每個樣本進行滑動窗口分割
    for i in range(samples):
        for j in range(num_windows):
            start_idx = j * stride
            end_idx = start_idx + window_size
            # 注意這裡是在 sequence_length 維度上進行切分
            windows[i * num_windows + j] = data[i, :, start_idx:end_idx]
    
    return windows

def prepare_data(features, labels, filenames, patient_ids, logger=None):
    """準備訓練、驗證和測試數據集
    
    Args:
        features: np.ndarray 特徵數據
        labels: np.ndarray 標籤數據
        filenames: List[str] 文件名列表
        patient_ids: List[str] 受試者ID列表
        logger: logging.Logger 日誌記錄器
    
    Returns:
        Tuple[
            Tuple[List[np.ndarray], List[int], List[str]],  # 訓練集
            Tuple[List[np.ndarray], List[int], List[str]],  # 驗證集
            Tuple[List[np.ndarray], List[int], List[str]],  # 測試集
            Tuple[List[str], List[str], List[str]]  # 受試者分組
        ]
    """
    if logger:
        logger.info("準備數據集...")
    
    # 確保 filenames 是字符串列表
    if isinstance(filenames[0], list):
        filenames = [f[0] for f in filenames]
    
    # 獲取唯一的受試者ID
    unique_patients = sorted(set(patient_ids))
    num_patients = len(unique_patients)
    
    # 計算每個集合的受試者數量
    num_train = int(num_patients * 0.7)
    num_val = int(num_patients * 0.15)
    
    # 隨機打亂受試者順序
    np.random.seed(42)
    shuffled_patients = np.random.permutation(unique_patients)
    
    # 分割受試者
    train_patients = shuffled_patients[:num_train]
    val_patients = shuffled_patients[num_train:num_train + num_val]
    test_patients = shuffled_patients[num_train + num_val:]
    
    # 根據受試者ID分割數據
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
    
    # 使用滑動窗口處理特徵
    def process_features_with_sliding_windows(indices):
        """使用滑動窗口處理特徵數據
        
        Args:
            indices: List[int] 數據索引列表
        
        Returns:
            Tuple[List[np.ndarray], List[int], List[str]] 處理後的特徵、標籤和文件名
        """
        processed_features = []
        processed_labels = []
        processed_filenames = []
        
        for idx in indices:
            # 獲取當前音頻文件的特徵
            feature = features[idx]  # [time_steps, feature_dim]
            
            # 使用滑動窗口分割特徵
            windows = []
            window_size = 129  # 約2.58秒
            stride = 64      # 50%重疊
            
            for start in range(0, len(feature) - window_size + 1, stride):
                end = start + window_size
                window = feature[start:end]
                windows.append(window)
            
            if windows:  # 確保有窗口
                windows = np.array(windows)  # [num_windows, window_size, feature_dim]
                processed_features.append(windows)
                processed_labels.append(labels[idx])
                processed_filenames.append(filenames[idx])
        
        return processed_features, processed_labels, processed_filenames
    
    # 處理訓練、驗證和測試集
    train_features, train_labels, train_filenames = process_features_with_sliding_windows(train_indices)
    val_features, val_labels, val_filenames = process_features_with_sliding_windows(val_indices)
    test_features, test_labels, test_filenames = process_features_with_sliding_windows(test_indices)
    
    if logger:
        logger.info(f"訓練集: {len(train_features)} 個音頻文件")
        logger.info(f"驗證集: {len(val_features)} 個音頻文件")
        logger.info(f"測試集: {len(test_features)} 個音頻文件")
        
        # 顯示每個音頻文件的窗口數量
        logger.info("\n訓練集窗口數量:")
        for i, (feat, fname) in enumerate(zip(train_features, train_filenames)):
            logger.info(f"  文件 {fname}: {len(feat)} 個窗口")
    
    # 返回處理後的數據集和受試者分組
    train_data = (train_features, train_labels, train_filenames)
    val_data = (val_features, val_labels, val_filenames)
    test_data = (test_features, test_labels, test_filenames)
    patient_splits = (list(train_patients), list(val_patients), list(test_patients))
    
    return train_data, val_data, test_data, patient_splits

def print_dataset_statistics(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str],
    patient_ids: List[str],
    logger: logging.Logger
):
    """打印詳細的數據集統計信息，包含受試者分布"""
    logger.info("\n=== 數據集分布統計 ===")
    
    # 打印受試者分布
    logger.info("\n受試者分布:")
    logger.info(f"訓練集受試者 ({len(train_patients)}): {', '.join(sorted(train_patients))}")
    logger.info(f"驗證集受試者 ({len(val_patients)}): {', '.join(sorted(val_patients))}")
    logger.info(f"測試集受試者 ({len(test_patients)}): {', '.join(sorted(test_patients))}")
    
    # 打印每個受試者的樣本數量
    logger.info("\n每個受試者的樣本數量:")
    for split_name, split_patients in [
        ("訓練集", train_patients),
        ("驗證集", val_patients),
        ("測試集", test_patients)
    ]:
        logger.info(f"\n{split_name}:")
        for patient in sorted(split_patients):
            samples = sum(1 for pid in patient_ids if pid == patient)
            logger.info(f"  受試者 {patient}: {samples} 樣本")
    
    # 打印類別分布
    logger.info("\n類別分布:")
    for split_name, labels in [
        ("訓練集", train_labels),
        ("驗證集", val_labels),
        ("測試集", test_labels)
    ]:
        class_dist = Counter(labels)
        logger.info(f"\n{split_name}:")
        for class_id, count in sorted(class_dist.items()):
            logger.info(f"  類別 {class_id}: {count} 樣本")
    logger.info("\n==================\n")

def save_split_info(
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str],
    save_dir: Path
):
    """保存數據分割信息，包含受試者ID和時間戳"""
    split_info = {
        'train_patients': sorted(train_patients),
        'val_patients': sorted(val_patients),
        'test_patients': sorted(test_patients),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 確保目錄存在
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存分割信息
    save_path = save_dir / 'split_info.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """訓練進度回調"""
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.logger.info(f"\n=== Epoch {epoch + 1} 統計 ===")
        self.logger.info(f"耗時: {epoch_time:.2f} 秒")
        self.logger.info(f"訓練損失: {logs['loss']:.4f}")
        self.logger.info(f"驗證損失: {logs['val_loss']:.4f}")
        self.logger.info(f"訓練準確率: {logs['accuracy']:.4f}")
        self.logger.info(f"驗證準確率: {logs['val_accuracy']:.4f}")
        self.logger.info("==================\n")

def setup_callbacks(save_dir: str) -> List[tf.keras.callbacks.Callback]:
    """設設置訓練回調函數

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
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'logs'),
            histogram_freq=1
        )
    ]
    return callbacks

def evaluate_model(model, features, labels, filenames, save_dir, logger):
    """評估模型性能
    
    Args:
        model: 模型實例
        features: List[np.ndarray] 每個音頻文件的所有窗口特徵
        labels: List[int] 每個音頻文件的標籤
        filenames: List[str] 每個音頻文件的文件名
        save_dir: str 保存目錄路徑
        logger: 日誌記錄器
    
    Returns:
        Tuple[float, np.ndarray, np.ndarray] 準確率、預測標籤、真實標籤
    """
    logger.info("開始評估模型...")
    
    # 初始化預測結果列表
    all_predictions = []
    all_true_labels = []
    
    # 對每個音頻文件進行預測
    for file_idx, (file_features, file_label, filename) in enumerate(zip(features, labels, filenames)):
        # 獲取該文件所有窗口的預測結果
        window_predictions = model.predict(file_features, verbose=0)  # [num_windows, num_classes]
        
        # 平均所有窗口的預測結果
        file_prediction = np.mean(window_predictions, axis=0)  # [num_classes]
        predicted_label = np.argmax(file_prediction)
        
        # 記錄預測結果
        all_predictions.append(predicted_label)
        all_true_labels.append(file_label)
        
        # 記錄詳細信息
        logger.info(f"文件 {filename}:")
        logger.info(f"  窗口數量: {len(file_features)}")
        logger.info(f"  真實標籤: {file_label}")
        logger.info(f"  預測標籤: {predicted_label}")
        logger.info(f"  預測正確: {'是' if predicted_label == file_label else '否'}")
    
    # 轉換為 numpy 數組
    predictions = np.array(all_predictions)
    true_labels = np.array(all_true_labels)
    
    # 計算準確率
    accuracy = np.mean(predictions == true_labels)
    
    # 計算並打印混淆矩陣
    cm = confusion_matrix(true_labels, predictions)
    logger.info("\n混淆矩陣:")
    logger.info(f"\n{cm}")
    
    # 保存混淆矩陣圖
    save_confusion_matrix(cm, save_dir)
    
    return accuracy, predictions, true_labels

class SequentialBatchGenerator(tf.keras.utils.Sequence):
    """順序批次生成器，保持同一音頻文件的窗口順序"""
    def __init__(self, features, labels, batch_size, shuffle=True):
        self.features = features  # 列表，每個元素是一個音頻文件的所有窗口
        self.labels = labels    # 列表，每個元素是一個音頻文件的標籤
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = get_num_classes()
        self.indices = np.arange(len(self.features))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """返回每個 epoch 的批次數量"""
        total_windows = sum(len(f) for f in self.features)
        return int(np.ceil(total_windows / self.batch_size))
    
    def __getitem__(self, idx):
        """獲取一個批次的數據"""
        batch_features = []
        batch_labels = []
        current_size = 0
        
        while current_size < self.batch_size:
            file_idx = self.indices[idx % len(self.indices)]
            file_features = self.features[file_idx]
            file_label = self.labels[file_idx]
            
            # 添加此文件的所有窗口
            for window in file_features:
                if current_size >= self.batch_size:
                    break
                # 移除額外的 channel 維度（如果存在）
                if len(window.shape) == 3:
                    window = np.squeeze(window, axis=-1)
                batch_features.append(window)
                batch_labels.append(file_label)
                current_size += 1
            
            if current_size < self.batch_size:
                idx = (idx + 1) % len(self.indices)
        
        # 將特徵轉換為 numpy 數組並確保形狀正確
        batch_features = np.array(batch_features)  # 形狀為 (batch_size, window_size, feature_dim)
        batch_labels = np.array(batch_labels)
        
        return batch_features, batch_labels
    
    def on_epoch_end(self):
        """每個 epoch 結束時重新打亂數據"""
        if self.shuffle:
            np.random.shuffle(self.indices)

def train_model(model, train_data, val_data, config, logger):
    """訓練模型
    
    Args:
        model: 模型實例
        train_data: 元組 (train_features, train_labels, train_filenames)
        val_data: 元組 (val_features, val_labels, val_filenames)
        config: 配置字典
        logger: 日誌記錄器
    
    Returns:
        history: 訓練歷史
        val_results: 驗證集評估結果
    """
    # 解包數據
    train_features, train_labels, train_filenames = train_data
    val_features, val_labels, val_filenames = val_data
    
    # 設置保存目錄
    save_dir = setup_save_dir()
    
    # 編譯模型
    logger.info("編譯模型...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 創建數據生成器
    train_generator = SequentialBatchGenerator(
        train_features,
        train_labels,
        config['training']['batch_size'],
        shuffle=True
    )
    
    val_generator = SequentialBatchGenerator(
        val_features,
        val_labels,
        config['training']['batch_size'],
        shuffle=False
    )
    
    # 設置回調函數
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.keras'),
            save_best_only=config['training'].get('save_best_only', True),
            monitor=config['training'].get('monitor_metric', 'val_loss'),
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            min_lr=config['training']['min_lr']
        ),
        TrainingProgressCallback(logger)
    ]
    
    # 訓練模型
    logger.info("開始訓練模型...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=config['training'].get('verbose', 1),
        use_multiprocessing=config['training'].get('use_multiprocessing', True),
        workers=config['training'].get('workers', 4)
    )
    
    # 評估驗證集
    logger.info("\n=== 驗證集評估 ===")
    val_results = evaluate_model(
        model,
        val_features,
        val_labels,
        val_filenames,
        save_dir,
        logger
    )
    logger.info(f"驗證集準確率: {val_results[0]:.4f}")
    
    return history, val_results

def load_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """加載數據並返回特徵和標籤

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]: 特徵、標籤、文件名和受試者ID
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data_dir']
    original_data_dir = config['original_data_dir']
    
    data_loader = AutoEncoderDataLoader(data_dir=data_dir, original_data_dir=original_data_dir)
    # 獲取特徵和標籤
    features, labels, filenames, patient_ids = data_loader.load_data(return_tokens=False)
    
    if len(features) == 0:
        raise ValueError("沒有找到有效的特徵數據")
    
    return features, labels, filenames, patient_ids

def main():
    """主函數"""
    # 設置日誌記錄器
    logger = setup_logger()
    
    try:
        # 讀取配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 打印類別配置
        print_class_config()
        
        # 加載數據
        data_loader = AutoEncoderDataLoader(
            data_dir=config['data_dir'],
            original_data_dir=config['original_data_dir']
        )
        features, labels, filenames, patient_ids = data_loader.load_data()
        logger.info(f"加載了 {len(features)} 個樣本")
        logger.info(f"合併後的特徵形狀: {features.shape}")
        logger.info(f"標籤形狀: {labels.shape}")
        logger.info(f"文件名數量: {len(filenames)}")
        logger.info(f"受試者數量: {len(set(patient_ids))}")
        
        # 準備數據集
        logger.info("準備數據集...")
        train_data, val_data, test_data, patient_splits = prepare_data(
            features,
            labels,
            filenames,
            patient_ids,
            logger=logger
        )
        
        # 解包數據
        train_features, train_labels, train_filenames = train_data
        val_features, val_labels, val_filenames = val_data
        test_features, test_labels, test_filenames = test_data
        train_patients, val_patients, test_patients = patient_splits
        
        # 打印詳細的數據集統計信息
        print_dataset_statistics(
            train_labels, val_labels, test_labels,
            train_patients, val_patients, test_patients,
            patient_ids, logger
        )
        
        # 設置保存目錄
        save_dir = Path(setup_save_dir())
        save_split_info(train_patients, val_patients, test_patients, save_dir)
        
        # 創建和編譯模型
        model = AutoencoderModel(config['model'])
        
        # 編譯模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 訓練模型
        history, val_results = train_model(
            model,
            train_data,
            val_data,
            config,
            logger
        )
        
        # 評估測試集
        logger.info("\n=== 測試集評估 ===")
        test_results = evaluate_model(
            model,
            test_features,
            test_labels,
            test_filenames,
            str(save_dir),
            logger
        )
        logger.info(f"測試集準確率: {test_results[0]:.4f}")
        
        # 保存訓練歷史
        save_history(history, str(save_dir))
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 