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
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """準備訓練、驗證和測試數據
    
    Args:
        features: 特徵數據，形狀為 [samples, channels, sequence_length]
        labels: 標籤數據
        filenames: 文件名列表
        patient_ids: 受試者ID列表
        logger: 日誌記錄器
        
    Returns:
        訓練集、驗證集和測試集的元組，每個元組包含特徵和標籤
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
    
    if logger:
        logger.info(f"原始特徵形狀: {features.shape}")
    
    # 使用滑動窗口分割數據
    window_size = 129  # 對應 example.py 中的 seq_num
    stride = 64       # window_size//2，確保50%重疊
    windows = sliding_window(features, window_size, stride)
    if logger:
        logger.info(f"滑動窗口分割後的特徵形狀: {windows.shape}")
    
    # 調整標籤以匹配窗口數量
    num_windows_per_sample = (features.shape[2] - window_size) // stride + 1
    window_labels = np.repeat(labels, num_windows_per_sample)
    
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
    
    # 根據受試者ID分割數據
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, pid in enumerate(patient_ids):
        start_idx = i * num_windows_per_sample
        end_idx = (i + 1) * num_windows_per_sample
        if pid in train_patients:
            train_indices.extend(range(start_idx, end_idx))
        elif pid in val_patients:
            val_indices.extend(range(start_idx, end_idx))
        else:
            test_indices.extend(range(start_idx, end_idx))
    
    # 創建數據集
    train_features = windows[train_indices]
    train_labels = window_labels[train_indices]
    val_features = windows[val_indices]
    val_labels = window_labels[val_indices]
    test_features = windows[test_indices]
    test_labels = window_labels[test_indices]
    
    # 打印分割信息
    if logger:
        logger.info(f"訓練集形狀: {train_features.shape}, 標籤形狀: {train_labels.shape}")
        logger.info(f"驗證集形狀: {val_features.shape}, 標籤形狀: {val_labels.shape}")
        logger.info(f"測試集形狀: {test_features.shape}, 標籤形狀: {test_labels.shape}")
    
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
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'logs'),
            histogram_freq=1
        )
    ]
    return callbacks

def evaluate_model(model, test_data, save_dir, logger):
    """評估模型性能
    Args:
        model: 訓練好的模型
        test_data: 測試數據字典，包含 'encoder_input' 和 'classifier_output'
        save_dir: 保存目錄
        logger: 日誌記錄器
    """
    # 計算測試集性能
    test_loss, test_accuracy = model.evaluate(
        test_data['encoder_input'],
        test_data['classifier_output']
    )
    logger.info(f'測試集準確率: {test_accuracy:.4f}')
    
    # 獲取預測結果
    y_pred = model.predict(test_data['encoder_input'])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_data['classifier_output'], axis=1)
    
    # 獲取類別名稱
    class_names = get_active_class_names()
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # 打印混淆矩陣（純文字格式）
    print('\n=== 混淆矩陣 ===')
    # 打印標題行
    print(f"{'':25}", end='')
    for name in class_names:
        print(f"{name:15}", end=' ')
    print('\n')
    
    # 打印每一行
    for i, (row, class_name) in enumerate(zip(cm, class_names)):
        print(f"{class_name:25}", end='')
        for val in row:
            print(f"{val:15}", end=' ')
        print()
    print('==================\n')
    
    # 保存混淆矩陣到文件
    confusion_matrix_file = os.path.join(save_dir, 'confusion_matrix.txt')
    with open(confusion_matrix_file, 'w', encoding='utf-8') as f:
        f.write('=== 混淆矩陣 ===\n')
        f.write(f"{'':25}")
        for name in class_names:
            f.write(f"{name:15} ")
        f.write('\n\n')
        
        for i, (row, class_name) in enumerate(zip(cm, class_names)):
            f.write(f"{class_name:25}")
            for val in row:
                f.write(f"{val:15} ")
            f.write('\n')
        f.write('==================\n')

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
        epochs=config.get('training', {}).get('epochs', 100),  # 從配置文件讀取 epochs
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
        (train_features, train_labels), (val_features, val_labels), (test_features, test_labels) = prepare_data(
            features,
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

        # 設置保存目錄
        save_dir = setup_save_dir()
        
        # 創建和編譯模型
        model = AutoencoderModel(config['model'])
        model.build()
        
        # 設置回調函數
        callbacks = setup_callbacks(save_dir)
        
        # 從配置文件獲取訓練參數
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        
        # 訓練模型
        history = model.fit(
            x=train_features,
            y=train_labels_onehot,
            validation_data=(val_features, val_labels_onehot),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # 評估模型並生成混淆矩陣
        test_data = {
            'encoder_input': test_features,
            'classifier_output': test_labels_onehot
        }
        evaluate_model(model.model, test_data, save_dir, logger)
        
        # 保存訓練歷史
        save_history(history, save_dir)
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        logger.error(f"Traceback:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 