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

# 添加項目根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    
    # 統計每個類別的樣本數
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

def prepare_data(
    features: np.ndarray,
    labels: np.ndarray,
    filenames: List[str],
    patient_ids: List[str],
    logger: Optional[logging.Logger] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[List[str], List[str], List[str]]]:
    """準備訓練、驗證和測試數據集，並返回受試者分組信息和文件索引
    
    Returns:
        Tuple containing:
        - (train_features, train_labels, train_file_indices)
        - (val_features, val_labels, val_file_indices)
        - (test_features, test_labels, test_file_indices)
        - (train_patients, val_patients, test_patients)
    """
    if logger:
        logger.info(f"原始特徵形狀: {features.shape}")
    
    # 確保 filenames 是字符串列表
    if isinstance(filenames[0], list):
        filenames = [f[0] if isinstance(f, list) else f for f in filenames]
    
    # 創建文件到索引的映射（使用基本文件名，不包含路徑）
    unique_files = sorted(set(os.path.basename(f) for f in filenames))
    file_to_idx = {file: idx for idx, file in enumerate(unique_files)}
    file_indices = np.array([file_to_idx[os.path.basename(f)] for f in filenames])
    
    # 獲取每個受試者的類別分布
    patient_class_dist = {}
    for pid, label in zip(patient_ids, labels):
        if pid not in patient_class_dist:
            patient_class_dist[pid] = []
        patient_class_dist[pid].append(label)
    
    # 按類別分布對受試者進行分層
    patient_groups = {i: [] for i in range(get_num_classes())}
    for pid, labels_list in patient_class_dist.items():
        main_class = Counter(labels_list).most_common(1)[0][0]
        patient_groups[main_class].append(pid)
    
    # 從每個類別中按比例選擇受試者
    train_patients, val_patients, test_patients = [], [], []
    for class_patients in patient_groups.values():
        if not class_patients:
            continue
        
        random.shuffle(class_patients)  # 隨機打亂每個類別的受試者
        train_size = int(len(class_patients) * 0.7)
        val_size = int(len(class_patients) * 0.15)
        
        train_patients.extend(class_patients[:train_size])
        val_patients.extend(class_patients[train_size:train_size + val_size])
        test_patients.extend(class_patients[train_size + val_size:])
    
    # 根據受試者ID分割數據
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
    
    def process_features_with_sliding_windows(features: np.ndarray, indices: List[int], file_indices: np.ndarray, window_size: int = 129, stride: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用滑動窗口處理特徵，並保持標籤和文件索引對應關係"""
        windows_list = []
        labels_list = []
        file_indices_list = []
        
        for idx in indices:
            feature = features[idx]
            # 計算可能的窗口數量
            num_windows = (feature.shape[1] - window_size) // stride + 1
            
            # 對每個樣本進行滑動窗口處理
            for i in range(num_windows):
                start = i * stride
                end = start + window_size
                window = feature[:, start:end]
                windows_list.append(window)
                labels_list.append(labels[idx])
                file_indices_list.append(file_indices[idx])
        
        return np.array(windows_list), np.array(labels_list), np.array(file_indices_list)
    
    # 使用滑動窗口處理每個數據集
    train_features, train_labels, train_file_indices = process_features_with_sliding_windows(features, train_indices, file_indices)
    val_features, val_labels, val_file_indices = process_features_with_sliding_windows(features, val_indices, file_indices)
    test_features, test_labels, test_file_indices = process_features_with_sliding_windows(features, test_indices, file_indices)
    
    if logger:
        # 計算原始音檔數量
        train_unique_files = len(set(file_indices[train_indices]))
        val_unique_files = len(set(file_indices[val_indices]))
        test_unique_files = len(set(file_indices[test_indices]))
        
        logger.info(f"訓練集：{train_unique_files} 個原始音檔, {len(train_features)} 個切片")
        logger.info(f"驗證集：{val_unique_files} 個原始音檔, {len(val_features)} 個切片")
        logger.info(f"測試集：{test_unique_files} 個原始音檔, {len(test_features)} 個切片")
    
    return (train_features, train_labels, train_file_indices), \
           (val_features, val_labels, val_file_indices), \
           (test_features, test_labels, test_file_indices), \
           (train_patients, val_patients, test_patients)

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
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'logs'),
            histogram_freq=1
        )
    ]
    return callbacks

def evaluate_model(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    file_indices: np.ndarray,
    save_dir: str,
    logger: logging.Logger
) -> Tuple[float, np.ndarray, np.ndarray]:
    """評估模型性能，將同一音檔的多個切片預測結果合併
    
    Args:
        model: 訓練好的模型
        features: 特徵數據
        labels: 標籤數據
        file_indices: 文件索引，用於識別同一音檔的不同切片
        save_dir: 保存目錄
        logger: 日誌記錄器
    """
    # 獲取每個切片的預測結果
    predictions = model.predict(features)
    
    # 按文件合併預測結果
    unique_files = np.unique(file_indices)
    merged_predictions = []
    merged_labels = []
    
    for file_idx in unique_files:
        # 獲取該文件的所有切片預測
        file_mask = file_indices == file_idx
        file_predictions = predictions[file_mask]
        
        # 對所有切片取平均得到最終預測
        mean_prediction = np.mean(file_predictions, axis=0)
        predicted_class = np.argmax(mean_prediction)
        
        # 獲取真實標籤（所有切片的標籤應該相同）
        true_label = labels[file_mask][0]
        
        merged_predictions.append(predicted_class)
        merged_labels.append(true_label)
    
    # 轉換為numpy數組
    merged_predictions = np.array(merged_predictions)
    merged_labels = np.array(merged_labels)
    
    # 計算準確率
    accuracy = np.mean(merged_predictions == merged_labels)
    logger.info(f'合併後的準確率: {accuracy:.4f} (基於 {len(unique_files)} 個原始音檔)')
    
    # 獲取類別名稱
    class_names = get_active_class_names()
    
    # 計算混淆矩陣
    cm = confusion_matrix(merged_labels, merged_predictions)
    
    # 打印混淆矩陣（純文字格式）
    logger.info('\n=== 混淆矩陣（基於原始音檔） ===')
    
    # 打印標題行
    header = '真實\\預測    ' + '    '.join(f'{name[:8]:8}' for name in class_names)
    logger.info(header)
    
    # 打印每一行
    for i, (row, class_name) in enumerate(zip(cm, class_names)):
        row_str = f'{class_name[:8]:8}    ' + '    '.join(f'{val:8d}' for val in row)
        logger.info(row_str)
    
    # 保存混淆矩陣到文件
    confusion_matrix_file = os.path.join(save_dir, 'confusion_matrix.txt')
    with open(confusion_matrix_file, 'w', encoding='utf-8') as f:
        f.write('=== 混淆矩陣（基於原始音檔） ===\n')
        f.write(header + '\n\n')
        for i, (row, class_name) in enumerate(zip(cm, class_names)):
            row_str = f'{class_name[:8]:8}    ' + '    '.join(f'{val:8d}' for val in row)
            f.write(row_str + '\n')
    
    return accuracy, merged_predictions, merged_labels

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
    save_dir = Path(config.get('training', {}).get('model_save_path', 'saved_models'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置回調
    callbacks = [
        # 模型檢查點
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / 'best_model.keras'),
            save_best_only=True,
            monitor=config['training']['monitor_metric'],
            mode='min'
        ),
        # 提前停止
        tf.keras.callbacks.EarlyStopping(
            monitor=config['training']['monitor_metric'],
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(save_dir / 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        ),
        # 自定義進度回調
        TrainingProgressCallback(logger)
    ]
    
    # 訓練模型
    history = model.fit(
        x=train_data['encoder_input'],
        y=train_data['classifier_output'],
        validation_data=(
            val_data['encoder_input'],
            val_data['classifier_output']
        ),
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=config['training']['verbose'],
        use_multiprocessing=config['training']['use_multiprocessing'],
        workers=config['training']['workers']
    )
    
    # 評估模型
    evaluation_results = evaluate_model(
        model,
        val_data,
        str(save_dir),
        logger
    )
    
    # 保存訓練歷史
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        json.dump(history_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"訓練歷史已保存到: {history_path}")
    
    return history.history, evaluation_results

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
        (train_features, train_labels, train_file_indices), \
        (val_features, val_labels, val_file_indices), \
        (test_features, test_labels, test_file_indices), \
        (train_patients, val_patients, test_patients) = prepare_data(
            features,
            labels,
            filenames,
            patient_ids,
            logger=logger
        )
        
        # 打印詳細的數據集統計信息
        print_dataset_statistics(
            train_labels, val_labels, test_labels,
            train_patients, val_patients, test_patients,
            patient_ids, logger
        )
        
        # 保存分割信息
        save_dir = Path(setup_save_dir())
        save_split_info(train_patients, val_patients, test_patients, save_dir)

        # 轉換標籤為 one-hot 編碼
        num_classes = get_num_classes()
        train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
        val_labels_onehot = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)
        test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
        
        # 創建和編譯模型
        model = AutoencoderModel(config['model'])
        model.build()
        
        # 設置回調函數
        callbacks = setup_callbacks(str(save_dir))
        
        # 訓練模型
        history = model.fit(
            x=train_features,
            y=train_labels_onehot,
            validation_data=(val_features, val_labels_onehot),
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            callbacks=callbacks,
            verbose=config['training']['verbose']
        )
        
        # 評估模型（基於原始文件）
        logger.info("\n=== 模型評估（基於原始文件） ===")
        
        # 評估訓練集
        train_acc, train_preds, train_true = evaluate_model(
            model, train_features, train_labels, train_file_indices,
            str(save_dir), logger
        )
        logger.info(f"訓練集準確率: {train_acc:.4f}")
        
        # 評估驗證集
        val_acc, val_preds, val_true = evaluate_model(
            model, val_features, val_labels, val_file_indices,
            str(save_dir), logger
        )
        logger.info(f"驗證集準確率: {val_acc:.4f}")
        
        # 評估測試集
        test_acc, test_preds, test_true = evaluate_model(
            model, test_features, test_labels, test_file_indices,
            str(save_dir), logger
        )
        logger.info(f"測試集準確率: {test_acc:.4f}")
        
        # 保存訓練歷史
        save_history(history, str(save_dir))
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 