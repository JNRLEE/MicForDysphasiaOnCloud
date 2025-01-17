"""
此代碼整合了WavFeatureCNN模型的所有組件，
包括配置、模型定義和訓練流程。
python Model/experiments/WavFeatureCNN/train_standalone.py
tensorboard --logdir=logs/ --port=6006
"""

import ostensorboard --logdir=logs/ --port=6006
import sys
import json
import numpy as np
import tensorflow as tf
import logging
import traceback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau
)

# 添加專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from collections import Counter
import time
from datetime import datetime
import glob
import re

# 從 class_config.py 導入必要的函數和配置
from Model.base.class_config import (
    is_normal,
    is_patient,
    get_action_type,
    is_class_active,
    get_class_mapping,
    CLASS_CONFIG,
    get_active_classes,
    get_num_classes
)

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 配置參數
CONFIG = {
    'feature_processing': {
        'window_size': 129,
        'target_dim': 2000,
        'stride': 64,
        'batch_proc_size': 32,
        'min_seq_length': 200,  # 最小序列長度
        'max_seq_length': 5000  # 最大序列長度
    },
    'dataset': {
        'data_dir': "WavData/CombinedData",
        'selection_types': {
            'NoMovement': 0,
            'DrySwallow': 1,
            'Cracker': 2,
            'Jelly': 3
        },
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'min_seq_length': 200,
        'max_seq_length': 2000
    },
    'audio': {
        'sr': 32000,
        'n_mels': 512,
        'hop_length': 512,
        'win_length': 2048,
        'window': 'hann',
        'center': True,
        'pad_mode': 'reflect'
    },
    'model': {
        'input_shape': (512, 2000),
        'conv_filters': [32, 64, 128],
        'conv_kernel_size': 3,
        'pool_size': 2,
        'dense_units': [256, 128],
        'dropout_rate': 0.5,
        'num_classes': 4
    },
    'training': {
        'batch_size': 8,  # 調整為較小的批次大小
        'epochs': 50,     # 增加訓練輪數
        'learning_rate': 0.001,
        'min_lr': 1e-6,
        'patience': 10,   # 增加早停的耐心值
        'factor': 0.5,
        'verbose': 1,
        'class_weights': None  # 將根據數據分布自動計算
    },
    'paths': {
        'save_dir': 'logs',
        'model_dir': 'logs/models',
        'tensorboard_dir': 'logs/tensorboard'
    }
}

# 確保日誌目錄存在
os.makedirs(CONFIG['paths']['save_dir'], exist_ok=True)
os.makedirs(CONFIG['paths']['model_dir'], exist_ok=True)
os.makedirs(CONFIG['paths']['tensorboard_dir'], exist_ok=True)

def prepare_label_mapping():
    """準備標籤映射"""
    label_mapping = {}
    for class_name, label in CLASS_CONFIG.items():
        subject_type, action = class_name.split('-')
        label_mapping[(subject_type, action)] = label
    return label_mapping

def process_features(features: np.ndarray, target_dim: int, logger: logging.Logger) -> np.ndarray:
    """處理特徵，使其符合目標維度
    
    Args:
        features: 輸入特徵，形狀為 (batch_size, time_steps, features) 或 (time_steps, features)
        target_dim: 目標時間步長
        logger: 日誌記錄器
    
    Returns:
        處理後的特徵
    """
    # 確保特徵是3D的
    if len(features.shape) == 2:
        features = np.expand_dims(features, axis=0)
    
    batch_size, time_steps, feature_dim = features.shape
    
    # 如果時間步長小於目標維度，使用填充
    if time_steps < target_dim:
        pad_width = ((0, 0), (0, target_dim - time_steps), (0, 0))
        features = np.pad(features, pad_width, mode='constant')
        logger.info(f"特徵已填充至目標維度 {target_dim}")
    # 如果時間步長大於目標維度，使用採樣
    elif time_steps > target_dim:
        indices = np.linspace(0, time_steps - 1, target_dim, dtype=int)
        features = features[:, indices, :]
        logger.info(f"特徵已採樣至目標維度 {target_dim}")
    
    return features

def determine_action_type(dirname: str, config: dict) -> Optional[str]:
    """從目錄名稱中確定動作類型
    
    Args:
        dirname: 目錄名稱
        config: 配置字典
    
    Returns:
        動作類型或None（如果無法確定）
    """
    for action, keywords in config['dataset']['selection_types'].items():
        if any(keyword in dirname for keyword in keywords):
            return action
    return None

def get_num_classes():
    """獲取活躍類別數量"""
    return sum(1 for v in CLASS_CONFIG.values() if v == 1)

def get_active_class_names():
    """獲取活躍類別名稱列表"""
    return [k for k, v in CLASS_CONFIG.items() if v == 1]

class TransposeLayer(tf.keras.layers.Layer):
    """轉置層，用於調整輸入特徵的維度順序"""
    def call(self, inputs):
        return tf.transpose(inputs, [0, 2, 1])

class WavFeatureCNN(tf.keras.Model):
    """簡化版的全連接層分類模型"""
    
    def __init__(self, config=None):
        """初始化模型
        
        Args:
            config: 配置字典，包含模型參數
        """
        super(WavFeatureCNN, self).__init__()
        
        # 轉置層
        self.transpose = TransposeLayer()
        
        # 展平層
        self.flatten = tf.keras.layers.Flatten()
        
        # 全連接層
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', dtype=tf.float32)
        self.dropout1 = tf.keras.layers.Dropout(0.3, dtype=tf.float32)
        
        self.dense2 = tf.keras.layers.Dense(32, activation='relu', dtype=tf.float32)
        self.dropout2 = tf.keras.layers.Dropout(0.3, dtype=tf.float32)
        
        # 輸出層
        self.output_dense = tf.keras.layers.Dense(4, dtype=tf.float32)  # 4個類別
        
    def call(self, inputs, training=False):
        """前向傳播
        
        Args:
            inputs: 輸入張量
            training: 是否處於訓練模式
            
        Returns:
            模型輸出
        """
        # 確保輸入是float32類型
        x = tf.cast(inputs, tf.float32)
        
        # 轉置處理
        x = self.transpose(x)
        
        # 展平
        x = self.flatten(x)
        
        # 全連接層
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        # 輸出層
        return self.output_dense(x)

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def build_model(self, input_shape):
        """構建模型"""
        self.build(input_shape)
        
        # 編譯模型
        self.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=CONFIG['training']['learning_rate'],
                clipnorm=1.0  # 添加梯度裁剪
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # 打印模型摘要
        self.summary()
        return self

def setup_logger() -> logging.Logger:
    """設置日誌記錄器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

def setup_save_dir() -> str:
    """設置保存目錄"""
    save_dir = os.path.join('saved_models', 'WavFeatureCNN')
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

def prepare_data(features, labels, patient_ids, test_size=0.2, val_size=0.2, logger=None):
    """
    準備訓練、驗證和測試數據集，確保同一個病人的資料只會出現在一個數據集中
    
    Args:
        features: 特徵數據
        labels: 標籤數據  
        patient_ids: 病人ID列表
        test_size: 測試集比例
        val_size: 驗證集比例
        logger: 日誌記錄器
    """
    
    # 獲取唯一的病人ID和對應的索引
    unique_patients = np.unique(patient_ids)
    patient_to_indices = {patient: [] for patient in unique_patients}
    
    # 為每個病人收集其所有數據的索引
    for idx, patient in enumerate(patient_ids):
        patient_to_indices[patient].append(idx)
    
    # 計算每個數據集需要的病人數量
    num_patients = len(unique_patients)
    num_test = max(1, int(num_patients * test_size))
    num_val = max(1, int(num_patients * val_size))
    num_train = num_patients - num_test - num_val
    
    if logger:
        logger.info(f"\n總病人數: {num_patients}")
        logger.info(f"訓練集病人數: {num_train}")
        logger.info(f"驗證集病人數: {num_val}")
        logger.info(f"測試集病人數: {num_test}")
    
    # 隨機打亂病人ID
    patient_list = list(unique_patients)
    random.shuffle(patient_list)
    
    # 分配病人到不同數據集
    train_patients = patient_list[:num_train]
    val_patients = patient_list[num_train:num_train+num_val]
    test_patients = patient_list[num_train+num_val:]
    
    # 收集每個數據集的索引
    train_indices = []
    val_indices = []
    test_indices = []
    
    for patient in train_patients:
        train_indices.extend(patient_to_indices[patient])
    for patient in val_patients:
        val_indices.extend(patient_to_indices[patient])
    for patient in test_patients:
        test_indices.extend(patient_to_indices[patient])
    
    # 分割數據
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    if logger:
        logger.info("\n=== 數據分割詳細信息 ===")
        logger.info(f"訓練集:")
        logger.info(f"  - 樣本數: {len(train_features)}")
        logger.info(f"  - 病人ID列表: {sorted(train_patients)}")
        logger.info(f"  - 每個病人的樣本數:")
        for p in sorted(train_patients):
            count = len([i for i in train_indices if patient_ids[i] == p])
            logger.info(f"    * {p}: {count}個樣本")
            
        logger.info(f"\n驗證集:")
        logger.info(f"  - 樣本數: {len(val_features)}")
        logger.info(f"  - 病人ID列表: {sorted(val_patients)}")
        logger.info(f"  - 每個病人的樣本數:")
        for p in sorted(val_patients):
            count = len([i for i in val_indices if patient_ids[i] == p])
            logger.info(f"    * {p}: {count}個樣本")
            
        logger.info(f"\n測試集:")
        logger.info(f"  - 樣本數: {len(test_features)}")
        logger.info(f"  - 病人ID列表: {sorted(test_patients)}")
        logger.info(f"  - 每個病人的樣本數:")
        for p in sorted(test_patients):
            count = len([i for i in test_indices if patient_ids[i] == p])
            logger.info(f"    * {p}: {count}個樣本")
        
        # 打印每個數據集中各類別的分布
        for dataset_name, dataset_labels in [
            ("訓練集", train_labels),
            ("驗證集", val_labels),
            ("測試集", test_labels)
        ]:
            unique, counts = np.unique(dataset_labels, return_counts=True)
            logger.info(f"\n{dataset_name}類別分布:")
            for label, count in zip(unique, counts):
                logger.info(f"類別 {label}: {count} 個樣本")
    
    return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)

def load_data(config, logger):
    """
    加載並準備數據
    """
    # 獲取所有特徵文件
    feature_files = glob.glob(os.path.join(config['dataset']['data_dir'], "**", "WavTokenizer_tokens.npy"), recursive=True)
    logger.info(f"找到 {len(feature_files)} 個特徵文件")
    
    # 初始化列表
    features_list = []
    labels_list = []
    filenames_list = []
    patient_ids_list = []
    
    # 獲取活躍類別的映射
    active_classes = {k: v for k, v in CLASS_CONFIG.items() if v == 1}
    class_mapping = {class_name: idx for idx, class_name in enumerate(active_classes.keys())}
    
    # 處理每個文件
    for file_path in feature_files:
        try:
            # 獲取目錄路徑
            dir_path = os.path.dirname(file_path)
            
            # 讀取 tokens_info.json
            info_path = os.path.join(dir_path, "WavTokenizer_tokens_info.json")
            if not os.path.exists(info_path):
                logger.warning(f"找不到特徵信息文件: {info_path}")
                continue
                
            with open(info_path, 'r', encoding='utf-8') as f:
                tokens_info = json.load(f)
            
            # 讀取 patient_info.json
            patient_info_files = glob.glob(os.path.join(dir_path, "*_info.json"))
            patient_info_files = [f for f in patient_info_files if not f.endswith("tokens_info.json")]
            
            if not patient_info_files:
                logger.warning(f"找不到病患信息文件: {dir_path}")
                continue
                
            with open(patient_info_files[0], 'r', encoding='utf-8') as f:
                patient_info = json.load(f)
            
            # 獲取病患資訊
            patient_id = patient_info.get('patientID', '')
            score = patient_info.get('score', -1)
            selection = patient_info.get('selection', '')
            
            # 使用 class_config.py 的函數判斷類別
            if is_normal(score):
                subject_type = 'Normal'
            elif is_patient(score):
                subject_type = 'Patient'
            else:
                logger.warning(f"無法確定受試者類型，評分: {score}")
                continue
            
            # 獲取動作類型
            action_type = get_action_type(selection)
            if not action_type:
                logger.warning(f"無法確定動作類型: {selection}")
                continue
            
            # 構建完整的類別名稱
            class_name = f"{subject_type}-{action_type}"
            
            # 檢查該類別是否在當前配置中激活
            if not is_class_active(class_name):
                logger.info(f"類別 {class_name} 未激活")
                continue
            
            # 獲取類別標籤
            if class_name not in class_mapping:
                logger.warning(f"類別 {class_name} 不在映射中")
                continue
            
            label = class_mapping[class_name]
            
            # 加載特徵數據
            data_dict = np.load(file_path, allow_pickle=True).item()
            features = data_dict['features']
            
            # 檢查特徵維度
            if len(features.shape) != 3 or features.shape[1] != 512:
                logger.warning(f"特徵維度不正確: {file_path}, shape={features.shape}")
                continue
            
            # 檢查序列長度
            if features.shape[2] < config['feature_processing']['min_seq_length']:
                logger.warning(f"序列長度太短: {file_path}, length={features.shape[2]}")
                continue
            if features.shape[2] > config['feature_processing']['max_seq_length']:
                logger.warning(f"序列長度太長: {file_path}, length={features.shape[2]}")
                continue
            
            # 處理特徵維度
            batch_size, time_steps, feature_dim = features.shape
            target_dim = config['feature_processing']['target_dim']
            
            if feature_dim > target_dim:
                # 如果特徵維度超過目標維度，從中間截取
                start = (feature_dim - target_dim) // 2
                features = features[:, :, start:start + target_dim]
            else:
                # 如果特徵維度小於目標維度，用0填充
                pad_width = ((0, 0), (0, 0), (0, target_dim - feature_dim))
                features = np.pad(features, pad_width, mode='constant', constant_values=0)
            
            # 添加到列表
            features_list.append(features)
            labels_list.extend([label] * len(features))
            filenames_list.extend([os.path.basename(dir_path)] * len(features))
            patient_ids_list.extend([patient_id] * len(features))
            
        except Exception as e:
            logger.error(f"處理文件時出錯 {file_path}: {str(e)}")
            continue
    
    if not features_list:
        raise ValueError("沒有找到有效的特徵文件")
    
    # 合併所有特徵
    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    
    # 特徵標準化
    features_reshaped = features.reshape(-1, features.shape[-1])
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_reshaped)
    features = features_normalized.reshape(features.shape)
    
    logger.info(f"數據加載完成:")
    logger.info(f"- 特徵形狀: {features.shape}")
    logger.info(f"- 標籤形狀: {labels.shape}")
    logger.info(f"- 樣本數量: {len(filenames_list)}")
    logger.info(f"- 受試者數量: {len(set(patient_ids_list))}")
    
    # 顯示每個類別的樣本數量
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        class_name = list(class_mapping.keys())[label]
        logger.info(f"- {class_name}: {count} 個樣本")
    
    # 分割數據
    train_data, val_data, test_data = prepare_data(
        features=features,
        labels=labels,
        patient_ids=patient_ids_list,
        test_size=config['dataset']['test_ratio'],
        val_size=config['dataset']['val_ratio'],
        logger=logger
    )
    
    # 記錄分割信息
    logger.info("\n=== 數據分割信息 ===")
    logger.info(f"訓練集:\n  - 樣本數: {len(train_data[0])}\n  - 病人數: {len(set(patient_ids_list))}")
    logger.info(f"驗證集:\n  - 樣本數: {len(val_data[0])}\n  - 病人數: {len(set(patient_ids_list))}")
    logger.info(f"測試集:\n  - 樣本數: {len(test_data[0])}\n  - 病人數: {len(set(patient_ids_list))}")
    
    # 顯示各集合的類別分布
    for name, data in [("訓練集", train_data), ("驗證集", val_data), ("測試集", test_data)]:
        logger.info(f"\n{name} 類別分布:")
        counts = Counter(data[1])
        for label, count in sorted(counts.items()):
            class_name = list(class_mapping.keys())[label]
            logger.info(f"  - {class_name}: {count}")
    
    return train_data, val_data, test_data, class_mapping

def evaluate_model(model, test_data, class_mapping, logger):
    """評估模型並生成混淆矩陣"""
    test_features, test_labels = test_data
    
    # 獲取預測結果
    predictions = model.predict(test_features)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # 計算準確率
    accuracy = accuracy_score(test_labels, predicted_labels)
    logger.info(f"\n測試集準確率: {accuracy:.4f}")
    
    # 計算混淆矩陣
    cm = confusion_matrix(test_labels, predicted_labels)
    
    # 計算其他指標
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision_score(test_labels, predicted_labels, average='macro')),
        'recall': float(recall_score(test_labels, predicted_labels, average='macro')),
        'f1': float(f1_score(test_labels, predicted_labels, average='macro')),
        'confusion_matrix': cm.tolist()
    }
    
    # 保存評估結果
    os.makedirs('logs', exist_ok=True)
    with open('logs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 繪製並保存混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('logs/confusion_matrix.png')
    plt.close()
    
    logger.info("\n評估結果已保存至: logs/evaluation_results.json")
    logger.info("混淆矩陣已保存至: logs/confusion_matrix.png")
    
    return results

def compute_class_weights(labels):
    """計算類別權重"""
    class_weights = {}
    for label in set(labels):
        class_weights[label] = 1 / np.sum(labels == label)
    return class_weights

def train_model(model, train_data, val_data, test_data, config):
    """訓練模型"""
    # 解包數據
    train_features, train_labels = train_data
    val_features, val_labels = val_data
    
    # 創建日誌目錄
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # 設置回調函數
    callbacks = [
        ModelCheckpoint(
            os.path.join(log_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(log_dir, "tensorboard"),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch"
        )
    ]
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    # 訓練模型
    history = model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=50,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存訓練歷史
    history_dict = history.history
    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(history_dict, f, indent=2)
    
    return history

def main():
    """主函數"""
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 啟用GPU記憶體增長
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("已啟用GPU記憶體增長")
        
        # 加載數據
        train_data, val_data, test_data, class_mapping = load_data(CONFIG, logger)
        
        # 構建模型
        logger.info("構建模型...")
        model = WavFeatureCNN(CONFIG)
        
        # 編譯模型
        logger.info("編譯模型...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # 訓練模型
        logger.info("開始訓練...")
        history = train_model(model, train_data, val_data, test_data, CONFIG)
        
        # 評估模型
        evaluate_model(model, test_data, class_mapping, logger)
        
    except Exception as e:
        logger.error(f"發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 