"""
此腳本用於訓練基線吞嚥聲分類模型
主要功能：
1. 加載和預處理頻譜特徵數據
2. 根據 class_config.py 動態配置分類類別
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
import matplotlib.pyplot as plt
import seaborn as sns

# 初始化 TPU
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("成功初始化 TPU")
    print("TPU 核心數量:", strategy.num_replicas_in_sync)
except ValueError:
    strategy = tf.distribute.get_strategy()
    print("未找到 TPU，使用默認策略:", strategy)
except Exception as e:
    print(f"TPU 初始化出錯: {str(e)}")
    strategy = tf.distribute.get_strategy()
    print("使用默認策略:", strategy)

# 取專案根目錄的絕對路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 切換到專案根目錄
os.chdir(PROJECT_ROOT)

# 添加項目根目錄到 Python 路徑
sys.path.append(PROJECT_ROOT)

from Model.data_loaders.spectrum_loader import SpectrumDataLoader
from Model.base.class_config import get_num_classes, get_active_class_names, CLASS_CONFIG
from Model.base.visualization import VisualizationTool
from Model.experiments.baseline.model import BaselineModel

def setup_logger() -> logging.Logger:
    """設置日誌記錄器"""
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
    save_dir = os.path.join('saved_models', 'baseline')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_history(history, save_dir: str):
    """保存訓練歷史"""
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

def prepare_data(
    features: np.ndarray,
    labels: np.ndarray,
    filenames: List[str],
    patient_ids: List[str],
    logger: Optional[logging.Logger] = None
) -> Tuple[Tuple, Tuple, Tuple, Tuple[List[str], List[str], List[str]]]:
    """準備訓練、驗證和測試數據集"""
    if logger:
        logger.info("準備數據集...")
    
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
    
    # 分割數據
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    train_filenames = [filenames[i] for i in train_indices]
    train_patient_ids = [patient_ids[i] for i in train_indices]
    
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    val_filenames = [filenames[i] for i in val_indices]
    val_patient_ids = [patient_ids[i] for i in val_indices]
    
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    test_filenames = [filenames[i] for i in test_indices]
    test_patient_ids = [patient_ids[i] for i in test_indices]
    
    if logger:
        logger.info(f"訓練集: {len(train_features)} 個樣本")
        logger.info(f"驗證集: {len(val_features)} 個樣本")
        logger.info(f"測試集: {len(test_features)} 個樣本")
    
    # 返回處理後的數據集和受試者分組
    train_data = (train_features, train_labels, train_filenames, train_patient_ids)
    val_data = (val_features, val_labels, val_filenames, val_patient_ids)
    test_data = (test_features, test_labels, test_filenames, test_patient_ids)
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
    """打印詳細的數據集統計信息"""
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
    class_names = get_active_class_names()
    for split_name, labels in [
        ("訓練集", train_labels),
        ("驗證集", val_labels),
        ("測試集", test_labels)
    ]:
        class_dist = Counter(np.argmax(labels, axis=1))
        logger.info(f"\n{split_name}:")
        for class_id, count in sorted(class_dist.items()):
            logger.info(f"  {class_names[class_id]}: {count} 樣本")
    logger.info("\n==================\n")

def save_split_info(
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str],
    save_dir: Path
):
    """保存數據分割信息"""
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
    """設置訓練回調函數"""
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
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    return callbacks

def save_confusion_matrix(cm: np.ndarray, save_dir: str, class_labels: Optional[List[str]] = None):
    """保存混淆矩陣圖像"""
    plt.figure(figsize=(12, 10))
    
    if class_labels is None:
        class_labels = [str(i) for i in range(cm.shape[0])]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    
    plt.title('混淆矩陣')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def evaluate_model(
    model: BaselineModel,
    features: np.ndarray,
    labels: np.ndarray,
    filenames: List[str],
    save_dir: str,
    logger: logging.Logger
) -> Dict:
    """評估模型性能"""
    logger.info("開始評估模型...")
    
    try:
        # 獲取預測結果
        predictions = model.model.predict(features)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)
        
        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # 獲取活躍的類別名稱
        class_names = get_active_class_names()
        
        # 顯示混淆矩陣
        logger.info("\n=== 混淆矩陣 ===")
        logger.info("\n真實標籤 (列) vs 預測標籤 (行):")
        
        # 打印列標籤
        header = "     "
        for i in range(len(class_names)):
            header += f"{i:>5}"
        logger.info(header)
        
        # 打印每一行
        for i, row in enumerate(cm):
            row_str = f"{i:>4}"
            for val in row:
                row_str += f"{val:>5}"
            logger.info(row_str)
        
        # 計算每個類別的指標
        logger.info("\n=== 各類別評估指標 ===")
        total_correct = 0
        total_samples = 0
        
        class_metrics = []
        for i, (name, row) in enumerate(zip(class_names, cm)):
            true_positive = row[i]
            total = np.sum(row)
            
            if total > 0:
                accuracy = true_positive / total
                total_correct += true_positive
                total_samples += total
                
                precision = true_positive / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
                recall = true_positive / total if total > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics.append({
                    'name': name,
                    'samples': total,
                    'correct': true_positive,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        # 按準確率排序並顯示結果
        class_metrics.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for metrics in class_metrics:
            logger.info(f"\n類別 {metrics['name']}:")
            logger.info(f"  樣本數: {metrics['samples']}")
            logger.info(f"  正確預測: {metrics['correct']}")
            logger.info(f"  準確率: {metrics['accuracy']:.4f}")
            logger.info(f"  精確率: {metrics['precision']:.4f}")
            logger.info(f"  召回率: {metrics['recall']:.4f}")
            logger.info(f"  F1分數: {metrics['f1']:.4f}")
        
        # 計算整體準確率
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        logger.info(f"\n整體準確率: {overall_accuracy:.4f}")
        
        # 保存混淆矩陣圖像
        save_confusion_matrix(cm, save_dir, class_names)
        
        return {
            'accuracy': overall_accuracy,
            'confusion_matrix': cm,
            'predictions': predicted_labels,
            'true_labels': true_labels,
            'class_metrics': class_metrics
        }
        
    except Exception as e:
        logger.error(f"評估過程中出錯: {str(e)}")
        logger.error("詳細錯誤:", exc_info=True)
        raise

def train_model(
    model: BaselineModel,
    train_data: Tuple,
    val_data: Tuple,
    config: Dict,
    logger: logging.Logger
) -> Tuple[tf.keras.callbacks.History, Dict]:
    """訓練模型"""
    # 解包數據
    train_features, train_labels, train_filenames, train_patient_ids = train_data
    val_features, val_labels, val_filenames, val_patient_ids = val_data
    
    # 設置回調函數
    callbacks = setup_callbacks(config.get('save_dir', 'saved_models/baseline'))
    callbacks.append(TrainingProgressCallback(logger))
    
    # 計算類別權重
    class_weights = None
    if config['training'].get('use_class_weights', True):
        true_labels = np.argmax(train_labels, axis=1)
        unique_labels = np.unique(true_labels)
        total_samples = len(true_labels)
        n_classes = len(unique_labels)
        
        class_weights = {}
        for label in unique_labels:
            class_count = np.sum(true_labels == label)
            weight = (1.0 / class_count) * (total_samples / n_classes)
            class_weights[label] = weight
        
        logger.info("\n=== 類別權重 ===")
        for label, weight in class_weights.items():
            logger.info(f"類別 {label}: {weight:.4f}")
    
    # 訓練模型
    logger.info("開始訓練模型...")
    history = model.train(
        (train_features, train_labels),
        (val_features, val_labels),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # 評估驗證集
    logger.info("\n=== 驗證集評估 ===")
    val_results = evaluate_model(
        model,
        val_features,
        val_labels,
        val_filenames,
        config.get('save_dir', 'saved_models/baseline'),
        logger
    )
    logger.info(f"驗證集準確率: {val_results['accuracy']:.4f}")
    
    return history, val_results

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
        data_loader = SpectrumDataLoader(
            data_dir=config['data_dir'],
            original_data_dir=config['original_data_dir']
        )
        features, labels, filenames, patient_ids = data_loader.load_data()
        logger.info(f"加載了 {len(features)} 個樣本")
        logger.info(f"特徵形狀: {features.shape}")
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
        train_features, train_labels, train_filenames, train_patient_ids = train_data
        val_features, val_labels, val_filenames, val_patient_ids = val_data
        test_features, test_labels, test_filenames, test_patient_ids = test_data
        train_patients, val_patients, test_patients = patient_splits
        
        # 打印詳細的數據集統計信息
        print_dataset_statistics(
            train_labels, val_labels, test_labels,
            train_patients, val_patients, test_patients,
            patient_ids, logger
        )
        
        # 設置保存目錄
        save_dir = Path(setup_save_dir())
        config['save_dir'] = str(save_dir)
        save_split_info(train_patients, val_patients, test_patients, save_dir)
        
        with strategy.scope():
            # 創建和編譯模型
            model = BaselineModel(config['model'])
            model.build()
        
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
        logger.info(f"測試集準確率: {test_results['accuracy']:.4f}")
        
        # 保存訓練歷史
        save_history(history, str(save_dir))
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 