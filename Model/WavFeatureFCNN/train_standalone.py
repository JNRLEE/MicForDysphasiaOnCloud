"""
此代碼整合了WavFeatureFCNN模型的所有組件，目前使用的是FC layer
包括配置、模型定義和訓練流程。
PYTHONPATH="/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia:$PYTHONPATH" python Model/WavFeatureFCNN/train_standalone.py
tensorboard --logdir=runs
rm -rf runs/*
"""

import os
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
from typing import Tuple, Dict, List, Optional, Set
from collections import Counter
import time
from datetime import datetime
import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from Model.base.class_config import (
    is_normal,
    is_patient,
    get_action_type,
    is_class_active,
    get_class_mapping,
    CLASS_CONFIG,
    get_active_classes,
    get_num_classes,
    subject_source,
    SUBJECT_SOURCE_CONFIG,
    SCORE_THRESHOLDS
)

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from visualize_tsne_custom import generate_tsne_plots

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
            'Jelly': 3,
            'WaterDrinking': 4
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
        'epochs': 100,     # 增加訓練輪數
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
    },
    'checkpoint_dir': 'checkpoints',  # 添加檢查點目錄
    'early_stopping_patience': 10,    # 添加早停的耐心值
}

# 確保日誌目錄存在
os.makedirs(CONFIG['paths']['save_dir'], exist_ok=True)
os.makedirs(CONFIG['paths']['model_dir'], exist_ok=True)
os.makedirs(CONFIG['paths']['tensorboard_dir'], exist_ok=True)

def get_run_dir() -> str:
    """
    獲取當前運行目錄，如果不存在則創建
    
    Returns:
        str: 運行目錄的路徑
    """
    if not hasattr(get_run_dir, '_run_dir'):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        get_run_dir._run_dir = os.path.join('runs', f'WavFeatureFCNN_{timestamp}')
        os.makedirs(get_run_dir._run_dir, exist_ok=True)
    return get_run_dir._run_dir

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
    """
    簡化版的全連接層分類模型，使用較小的網絡容量和穩定的訓練策略
    """
    def __init__(self, config=None):
        """
        初始化模型，設置所有層
        """
        super().__init__()
        self.transpose = TransposeLayer()
        self.flatten = tf.keras.layers.Flatten()
        
        # 第一個全連接層組
        self.dense1 = tf.keras.layers.Dense(64, name='dense1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.act1 = tf.keras.layers.ReLU(name='act1')
        self.drop1 = tf.keras.layers.Dropout(0.3, name='drop1')
        
        # 輸出層
        num_classes = get_num_classes()  # 從配置中獲取類別數
        self.output_dense = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')

    def call(self, inputs, training=False):
        """
        前向傳播
        """
        x = tf.cast(inputs, tf.float32)
        x = self.transpose(x)
        x = self.flatten(x)
        
        # 第一個全連接層組
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        
        # 輸出層
        return self.output_dense(x)

    def get_config(self):
        return super().get_config()

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

def categorize_subjects(patient_ids: List[str], scores: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """
    將受試者分類為正常人和病人
    
    Args:
        patient_ids: 所有受試者ID列表
        scores: 受試者ID到評分的映射字典
        
    Returns:
        Tuple[List[str], List[str]]: 正常人ID列表和病人ID列表
    """
    normal_subjects = []
    patient_subjects = []
    
    for pid in patient_ids:
        if pid in scores:
            if is_normal(scores[pid]):
                normal_subjects.append(pid)
            elif is_patient(scores[pid]):
                patient_subjects.append(pid)
                
    return normal_subjects, patient_subjects

def save_distribution_info(log_dir: str, train_info: Dict, val_info: Dict, test_info: Dict):
    """
    保存數據分布信息到markdown文件
    
    Args:
        log_dir: 日誌目錄
        train_info: 訓練集信息
        val_info: 驗證集信息
        test_info: 測試集信息
    """
    # 確保目錄存在
    os.makedirs(log_dir, exist_ok=True)
    
    md_path = os.path.join(log_dir, "DataDistribution.md")
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 數據分布詳細信息\n\n")
        
        # 添加配置信息區段
        f.write("## 配置信息\n\n")
        
        # 記錄評分閾值設定
        f.write("### 評分閾值設定\n")
        f.write(f"- 正常人閾值: <= {SCORE_THRESHOLDS['normal']}\n")
        f.write(f"- 病人閾值: >= {SCORE_THRESHOLDS['patient']}\n\n")
        
        # 記錄來源設定
        f.write("### 來源設定\n")
        f.write("#### 正常人來源設定\n")
        f.write(f"- N開頭: {'包含' if SUBJECT_SOURCE_CONFIG['normal']['include_N'] else '不包含'}\n")
        f.write(f"- P開頭: {'包含' if SUBJECT_SOURCE_CONFIG['normal']['include_P'] else '不包含'}\n\n")
        f.write("#### 病人來源設定\n")
        f.write(f"- N開頭: {'包含' if SUBJECT_SOURCE_CONFIG['patient']['include_N'] else '不包含'}\n")
        f.write(f"- P開頭: {'包含' if SUBJECT_SOURCE_CONFIG['patient']['include_P'] else '不包含'}\n\n")
        
        # 記錄類別配置
        f.write("### 類別配置\n")
        for class_name, is_active in CLASS_CONFIG.items():
            status = "啟用" if is_active == 1 else "停用"
            f.write(f"- {class_name}: {status}\n")
        f.write("\n")
        
        # 數據集分布信息
        for dataset_name, info in [
            ("訓練集", train_info),
            ("驗證集", val_info),
            ("測試集", test_info)
        ]:
            f.write(f"## {dataset_name}\n\n")
            f.write(f"### 樣本數: {info['sample_count']}\n\n")
            
            f.write("### 受試者分布\n")
            f.write(f"- 正常人數量: {len(info['normal_subjects'])}\n")
            f.write("  - ID列表: " + ", ".join(sorted(info['normal_subjects'])) + "\n")
            f.write(f"- 病人數量: {len(info['patient_subjects'])}\n")
            f.write("  - ID列表: " + ", ".join(sorted(info['patient_subjects'])))
            f.write("\n\n")
            
            f.write("### 每個受試者的樣本數\n")
            for pid, count in sorted(info['subject_samples'].items()):
                subject_type = "正常人" if pid in info['normal_subjects'] else "病人"
                f.write(f"- {pid} ({subject_type}): {count}個樣本\n")
            f.write("\n")
            
            f.write("### 類別分布\n")
            for label, count in sorted(info['class_distribution'].items()):
                f.write(f"- 類別 {label}: {count}個樣本\n")
            f.write("\n")

def prepare_data(features, labels, patient_ids, file_paths, test_size=0.2, val_size=0.2, logger=None, patient_scores=None):
    """
    準備訓練、驗證和測試數據集，確保同一個病人的資料只會出現在一個數據集中
    """
    # 獲取唯一的病人ID和對應的索引
    unique_patients = np.unique(patient_ids)
    patient_to_indices = {patient: [] for patient in unique_patients}
    
    # 為每個病人收集其所有數據的索引
    for idx, patient in enumerate(patient_ids):
        patient_to_indices[patient].append(idx)
    
    # 將受試者分類為正常人和病人
    normal_subjects = []
    patient_subjects = []
    
    for pid in unique_patients:
        if pid in patient_scores:
            is_normal_subject, is_patient_subject = subject_source(patient_scores[pid], pid)
            if is_normal_subject:
                normal_subjects.append(pid)
            elif is_patient_subject:
                patient_subjects.append(pid)
    
    # 計算每個數據集需要的病人數量
    num_patients = len(unique_patients)
    num_test = max(1, int(num_patients * test_size))
    num_val = max(1, int(num_patients * val_size))
    num_train = num_patients - num_test - num_val
    
    if logger:
        logger.info(f"\n總受試者數: {num_patients}")
        logger.info(f"- 正常人: {len(normal_subjects)}人")
        logger.info(f"- 病人: {len(patient_subjects)}人")
        logger.info(f"\n數據集分配:")
        logger.info(f"- 訓練集: {num_train}人")
        logger.info(f"- 驗證集: {num_val}人")
        logger.info(f"- 測試集: {num_test}人")
    
    # 分別打亂正常人和病人列表
    random.shuffle(normal_subjects)
    random.shuffle(patient_subjects)
    
    # 按比例分配正常人和病人到各數據集
    normal_train = normal_subjects[:int(len(normal_subjects) * 0.6)]
    normal_val = normal_subjects[int(len(normal_subjects) * 0.6):int(len(normal_subjects) * 0.8)]
    normal_test = normal_subjects[int(len(normal_subjects) * 0.8):]
    
    patient_train = patient_subjects[:int(len(patient_subjects) * 0.6)]
    patient_val = patient_subjects[int(len(patient_subjects) * 0.6):int(len(patient_subjects) * 0.8)]
    patient_test = patient_subjects[int(len(patient_subjects) * 0.8):]
    
    # 合併各數據集的受試者
    train_patients = normal_train + patient_train
    val_patients = normal_val + patient_val
    test_patients = normal_test + patient_test
    
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
    train_paths = [file_paths[i] for i in train_indices]
    
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    val_paths = [file_paths[i] for i in val_indices]
    
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    test_paths = [file_paths[i] for i in test_indices]
    
    # 準備數據分布信息
    datasets_info = {
        'train': {
            'sample_count': len(train_features),
            'normal_subjects': normal_train,
            'patient_subjects': patient_train,
            'subject_samples': {p: len([i for i in train_indices if patient_ids[i] == p]) 
                              for p in train_patients},
            'class_distribution': dict(Counter(train_labels))
        },
        'val': {
            'sample_count': len(val_features),
            'normal_subjects': normal_val,
            'patient_subjects': patient_val,
            'subject_samples': {p: len([i for i in val_indices if patient_ids[i] == p]) 
                              for p in val_patients},
            'class_distribution': dict(Counter(val_labels))
        },
        'test': {
            'sample_count': len(test_features),
            'normal_subjects': normal_test,
            'patient_subjects': patient_test,
            'subject_samples': {p: len([i for i in test_indices if patient_ids[i] == p]) 
                              for p in test_patients},
            'class_distribution': dict(Counter(test_labels))
        }
    }
    
    if logger:
        logger.info("\n=== 數據分割詳細信息 ===")
        for dataset_name, info in datasets_info.items():
            logger.info(f"\n{dataset_name.capitalize()}集:")
            logger.info(f"  - 樣本數: {info['sample_count']}")
            logger.info(f"  - 正常人ID列表: {sorted(info['normal_subjects'])}")
            logger.info(f"  - 病人ID列表: {sorted(info['patient_subjects'])}")
            logger.info(f"  - 每個受試者的樣本數:")
            for p in sorted(info['subject_samples'].keys()):
                subject_type = "正常人" if p in info['normal_subjects'] else "病人"
                logger.info(f"    * {p} ({subject_type}): {info['subject_samples'][p]}個樣本")
            
            logger.info(f"  - 類別分布:")
            for label, count in sorted(info['class_distribution'].items()):
                logger.info(f"    * 類別 {label}: {count}個樣本")
    
    return (train_features, train_labels, train_paths), \
           (val_features, val_labels, val_paths), \
           (test_features, test_labels, test_paths), \
           datasets_info

def load_data(config, logger):
    """加載並準備數據"""
    # 顯示受試者分類配置
    logger.info("\n=== 受試者分類配置 ===")
    logger.info("正常人定義:")
    logger.info("- 評分閾值: <= 0")
    logger.info("- 來源設定:")
    logger.info(f"  * N開頭: {'包含' if SUBJECT_SOURCE_CONFIG['normal']['include_N'] else '不包含'}")
    logger.info(f"  * P開頭: {'包含' if SUBJECT_SOURCE_CONFIG['normal']['include_P'] else '不包含'}")
    logger.info("\n病人定義:")
    logger.info("- 評分閾值: >= 10")
    logger.info("- 來源設定:")
    logger.info(f"  * N開頭: {'包含' if SUBJECT_SOURCE_CONFIG['patient']['include_N'] else '不包含'}")
    logger.info(f"  * P開頭: {'包含' if SUBJECT_SOURCE_CONFIG['patient']['include_P'] else '不包含'}\n")

    # 獲取所有特徵文件
    feature_files = glob.glob(os.path.join(config['dataset']['data_dir'], "**", "WavTokenizer_tokens.npy"), recursive=True)
    logger.info(f"找到 {len(feature_files)} 個特徵文件")
    
    # 初始化列表和字典
    features_list = []
    labels_list = []
    filenames_list = []
    patient_ids_list = []
    file_paths_list = []
    patient_scores = {}
    
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
            
            # 使用新的 subject_source 函數判斷類別
            is_normal_subject, is_patient_subject = subject_source(score, patient_id)
            
            if is_normal_subject:
                subject_type = 'Normal'
            elif is_patient_subject:
                subject_type = 'Patient'
            else:
                logger.warning(f"無法確定受試者類型，ID: {patient_id}, 評分: {score}")
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
                start = (feature_dim - target_dim) // 2
                features = features[:, :, start:start + target_dim]
            else:
                pad_width = ((0, 0), (0, 0), (0, target_dim - feature_dim))
                features = np.pad(features, pad_width, mode='constant', constant_values=0)
            
            # 添加到列表
            features_list.append(features)
            labels_list.extend([label] * len(features))
            filenames_list.extend([os.path.basename(dir_path)] * len(features))
            patient_ids_list.extend([patient_id] * len(features))
            
            # 保存對應的音頻文件路徑
            wav_file = os.path.join(dir_path, "Probe0_RX_IN_TDM4CH0.wav")
            if os.path.exists(wav_file):
                rel_path = os.path.relpath(wav_file, config['dataset']['data_dir'])
                file_paths_list.extend([rel_path] * len(features))
            else:
                logger.warning(f"找不到音頻文件: {wav_file}")
                file_paths_list.extend([os.path.basename(dir_path)] * len(features))
            
            # 保存病人評分
            patient_scores[patient_id] = score
            
        except Exception as e:
            logger.error(f"處理文件時出錯 {file_path}: {str(e)}")
            continue
    
    if not features_list:
        raise ValueError("沒有找到有效的特徵文件")
    
    # 合併所有特徵
    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    file_paths = np.array(file_paths_list)
    
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
    
    # 分割數據，傳入病人評分信息
    train_data, val_data, test_data, datasets_info = prepare_data(
        features=features,
        labels=labels,
        patient_ids=patient_ids_list,
        file_paths=file_paths_list,
        test_size=config['dataset']['test_ratio'],
        val_size=config['dataset']['val_ratio'],
        logger=logger,
        patient_scores=patient_scores
    )
    
    # 保存分布信息到文件
    save_distribution_info(
        log_dir=get_run_dir(),
        train_info=datasets_info['train'],
        val_info=datasets_info['val'],
        test_info=datasets_info['test']
    )
    
    return train_data, val_data, test_data, class_mapping, file_paths_list

def evaluate_model(model, test_data, class_mapping, logger, run_dir: str):
    """
    評估模型在測試集上的表現
    
    Args:
        model: 訓練好的模型
        test_data: 測試數據集
        class_mapping: 類別映射
        logger: 日誌記錄器
        run_dir: 運行目錄路徑
    """
    test_features, test_labels, test_paths = test_data
    predictions = model.predict(test_features)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_labels if len(test_labels.shape) == 1 else np.argmax(test_labels, axis=1)

    # 計算準確率
    accuracy = accuracy_score(true_labels, predicted_labels)
    logger.info(f"\n測試集準確率: {accuracy:.4f}")

    # 計算混淆矩陣
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 計算每個類別的精確度、召回率和F1分數
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)

    # 創建預測結果列表
    prediction_results = []
    for i in range(len(test_paths)):
        result = {
            "file_path": test_paths[i],
            "true_label": int(true_labels[i]),
            "predicted_label": int(predicted_labels[i]),
            "correct": bool(true_labels[i] == predicted_labels[i])
        }
        prediction_results.append(result)

    # 保存預測結果到 JSON 文件
    prediction_results_path = os.path.join(run_dir, "prediction_results.json")
    with open(prediction_results_path, "w", encoding="utf-8") as f:
        json.dump(prediction_results, f, indent=4, ensure_ascii=False)

    # 保存混淆矩陣到 CSV 文件
    cm_df = pd.DataFrame(cm)
    cm_csv_path = os.path.join(run_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)

    # 獲取類別名稱列表
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    
    # 繪製並保存混淆矩陣熱圖
    cm_plot_path = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    # 保存評估指標
    metrics = {
        "accuracy": float(accuracy),
        "precision": {class_names[i]: float(p) for i, p in enumerate(precision)},
        "recall": {class_names[i]: float(r) for i, r in enumerate(recall)},
        "f1": {class_names[i]: float(f) for i, f in enumerate(f1)}
    }
    
    metrics_path = os.path.join(run_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    logger.info(f"評估結果已保存到目錄: {run_dir}")

def compute_class_weights(labels):
    """計算類別權重"""
    class_weights = {}
    for label in set(labels):
        class_weights[label] = 1 / np.sum(labels == label)
    return class_weights

class BatchLossCallback(tf.keras.callbacks.Callback):
    """用於記錄每個批次的訓練指標的回調函數"""
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        self.current_epoch = 0
        self.current_batch = 0
        self.batch_metrics = {}
        self.total_batches = 0
        
    def on_train_begin(self, logs=None):
        """訓練開始時初始化TensorBoard writer"""
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, 'batch_metrics')
        )
        
    def on_epoch_begin(self, epoch, logs=None):
        """每個epoch開始時更新當前epoch計數"""
        self.current_epoch = epoch
        self.current_batch = 0
        self.batch_metrics = {}
        
    def on_train_batch_end(self, batch, logs=None):
        """每個batch結束時記錄指標"""
        if logs is not None and self.writer is not None:
            with self.writer.as_default():
                # 為每個batch創建獨立的指標
                for metric_name, metric_value in logs.items():
                    if isinstance(metric_value, (int, float)):
                        # 使用total_batches作為x軸
                        self.total_batches += 1
                        
                        # 記錄每個batch的原始值
                        tf.summary.scalar(
                            f'batch_{metric_name}/batch_{self.current_batch}',
                            metric_value,
                            step=self.total_batches
                        )
                        
                        # 記錄移動平均
                        if metric_name not in self.batch_metrics:
                            self.batch_metrics[metric_name] = []
                        self.batch_metrics[metric_name].append(metric_value)
                        window_size = min(50, len(self.batch_metrics[metric_name]))
                        moving_avg = np.mean(self.batch_metrics[metric_name][-window_size:])
                        tf.summary.scalar(
                            f'batch_{metric_name}/moving_average',
                            moving_avg,
                            step=self.total_batches
                        )
                
                self.writer.flush()
            self.current_batch += 1

class AutoMonitor(tf.keras.callbacks.Callback):
    """自動監控訓練狀況的回調函數"""
    
    def __init__(self, patience=5, min_delta=0.001, check_interval=100):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.check_interval = check_interval
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.losses = []
        self.batch_count = 0
        
    def on_train_batch_end(self, batch, logs=None):
        """每個batch結束時檢查訓練狀況"""
        current_loss = logs.get('loss')
        if current_loss is not None:
            self.losses.append(current_loss)
            self.batch_count += 1
            
            if self.batch_count % self.check_interval == 0:
                # 計算最近check_interval個batch的平均損失
                recent_avg_loss = np.mean(self.losses[-self.check_interval:])
                
                if recent_avg_loss < self.best_loss - self.min_delta:
                    self.best_loss = recent_avg_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    print(f'\n自動停止訓練: 最近{self.check_interval}個batch的平均損失沒有顯著改善')
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'訓練在第 {self.stopped_epoch} 個epoch自動停止')

class BatchLogger(tf.keras.callbacks.Callback):
    """記錄每個batch的詳細資訊的回調函數"""
    def __init__(self, log_dir, file_paths):
        super().__init__()
        self.log_dir = log_dir
        self.file_paths = file_paths
        self.batch_logs = []
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'batch_metrics'))
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.batch_size = 4  # 設置默認的batch_size
        
    def on_epoch_begin(self, epoch, logs=None):
        self.batch_logs = []
        self.current_epoch = epoch
        self.current_batch = 0
        print(f"\n=== 開始第 {epoch + 1} 個訓練週期 ===")
        
    def on_train_batch_begin(self, batch, logs=None):
        """在每個batch開始前打印文件信息"""
        start_idx = batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.file_paths))
        
        print(f"\n=== Batch {self.current_batch + 1} 訓練文件 ===")
        print(f"Epoch {self.current_epoch + 1}, Batch {self.current_batch + 1}")
        print(f"樣本範圍: {start_idx} - {end_idx}")
        for idx in range(start_idx, end_idx):
            if idx < len(self.file_paths):
                file_path = self.file_paths[idx]
                print(f"- {file_path}")
        print("========================\n")
        
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', self.batch_size)
        
        # 計算當前批次的樣本索引
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, len(self.file_paths))
        
        # 獲取當前批次的文件路徑
        batch_files = self.file_paths[start_idx:end_idx] if start_idx < len(self.file_paths) else []
        
        # 記錄批次信息
        batch_info = {
            'epoch': self.current_epoch + 1,
            'batch': self.current_batch + 1,
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
            'files': batch_files.tolist() if isinstance(batch_files, np.ndarray) else batch_files
        }
        self.batch_logs.append(batch_info)
        
        # 將每個batch的指標寫入TensorBoard
        with self.writer.as_default():
            self.total_batches += 1
            for metric_name, metric_value in logs.items():
                if isinstance(metric_value, (int, float)):
                    tf.summary.scalar(
                        f'batch_metrics/{metric_name}/batch_{self.current_batch}',
                        metric_value,
                        step=self.total_batches
                    )
        
        # 打印訓練指標
        print(f"Batch {self.current_batch + 1} 訓練結果:")
        print(f"- Loss: {batch_info['loss']:.4f}")
        print(f"- Accuracy: {batch_info['accuracy']:.4f}")
        print("------------------------\n")
        
        self.current_batch += 1
            
    def on_epoch_end(self, epoch, logs=None):
        # 將批次日誌保存到文件
        log_file = os.path.join(self.log_dir, f'batch_logs_epoch_{epoch+1}.json')
        with open(log_file, 'w') as f:
            json.dump(self.batch_logs, f, indent=2)
        print(f"\n=== 第 {epoch + 1} 個訓練週期結束 ===")
        print(f"批次日誌已保存至: {log_file}\n")

def train_model(model, train_data, val_data, config):
    """訓練模型"""
    # 啟用記憶體優化
    tf.config.experimental.enable_tensor_float_32_execution(True)
    
    # 設置較小的批次大小
    batch_size = 4
    
    # 獲取運行目錄
    run_dir = get_run_dir()
    
    # 創建batch日誌目錄
    batch_log_dir = os.path.join(run_dir, "batch_logs")
    os.makedirs(batch_log_dir, exist_ok=True)
    
    # 保存訓練配置
    config_path = os.path.join(run_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 將元組數據轉換為tf.data.Dataset
    train_features, train_labels, train_paths = train_data
    val_features, val_labels, val_paths = val_data
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    
    # 設置數據集批次和預取
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 設置回調函數
    callbacks = [
        BatchLossCallback(run_dir),
        BatchLogger(batch_log_dir, train_paths),
        AutoMonitor(patience=5, min_delta=0.001, check_interval=100),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['factor'],
            patience=5,
            min_lr=config['training']['min_lr']
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=run_dir,
            histogram_freq=0,
            write_images=False,
            update_freq='batch'
        )
    ]
    
    # 訓練模型
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_config_info(run_dir: str):
    """
    保存配置信息到運行目錄
    
    Args:
        run_dir: 運行目錄路徑
    """
    config_info = {
        "score_thresholds": SCORE_THRESHOLDS,
        "subject_source_config": SUBJECT_SOURCE_CONFIG,
        "class_config": CLASS_CONFIG,
        "model_config": CONFIG
    }
    
    config_path = os.path.join(run_dir, "config_info.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str], 
                         save_path: str,
                         title: str = 'Confusion Matrix'):
    """
    繪製並保存混淆矩陣熱圖
    
    Args:
        cm: 混淆矩陣數據
        class_names: 類別名稱列表
        save_path: 保存路徑
        title: 圖表標題
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_color_mapping(train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]) -> Dict[str, str]:
    """
    生成顏色映射字典
    
    Args:
        train_ids: 訓練集病人ID集合
        val_ids: 驗證集病人ID集合
        test_ids: 測試集病人ID集合
        
    Returns:
        Dict[str, str]: 顏色映射字典
    """
    # 基礎顏色系統 (10種不同色系)
    base_colors = [
        '#FF0000',  # 紅
        '#00FF00',  # 綠
        '#0000FF',  # 藍
        '#FF00FF',  # 紫
        '#FFFF00',  # 黃
        '#00FFFF',  # 青
        '#FFA500',  # 橙
        '#800080',  # 深紫
        '#008000',  # 深綠
        '#800000'   # 深紅
    ]
    
    color_map = {}
    active_classes = [cls for cls, active in CLASS_CONFIG.items() if active == 1]
    
    # 為每個活動類別分配三種深淺度的顏色
    for idx, class_name in enumerate(active_classes):
        base_color = base_colors[idx % len(base_colors)]
        # 轉換為RGB
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        # 生成三種深淺度
        for patient_id in train_ids:
            color_map[f"{class_name}_{patient_id}_train"] = f"#{r:02x}{g:02x}{b:02x}"
        for patient_id in val_ids:
            darker_rgb = (int(r*0.7), int(g*0.7), int(b*0.7))
            color_map[f"{class_name}_{patient_id}_val"] = f"#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}"
        for patient_id in test_ids:
            darkest_rgb = (int(r*0.4), int(g*0.4), int(b*0.4))
            color_map[f"{class_name}_{patient_id}_test"] = f"#{darkest_rgb[0]:02x}{darkest_rgb[1]:02x}{darkest_rgb[2]:02x}"
    
    return color_map

def plot_tsne_visualization(tsne_results: pd.DataFrame, 
                          run_dir: str,
                          train_ids: Set[str],
                          val_ids: Set[str],
                          test_ids: Set[str],
                          perplexity: int,
                          is_3d: bool = False):
    """
    繪製TSNE視覺化圖
    
    Args:
        tsne_results: TSNE結果DataFrame
        run_dir: 運行目錄
        train_ids: 訓練集病人ID集合
        val_ids: 驗證集病人ID集合
        test_ids: 測試集病人ID集合
        perplexity: TSNE的perplexity參數
        is_3d: 是否為3D圖
    """
    color_map = generate_color_mapping(train_ids, val_ids, test_ids)
    
    plt.figure(figsize=(12, 8))
    if is_3d:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    
    # 獲取對應的座標列
    if is_3d:
        x_col = f'tsne_3d_p{perplexity}_x'
        y_col = f'tsne_3d_p{perplexity}_y'
        z_col = f'tsne_3d_p{perplexity}_z'
    else:
        x_col = f'tsne_2d_p{perplexity}_x'
        y_col = f'tsne_2d_p{perplexity}_y'
    
    # 繪製每個點
    for _, row in tsne_results.iterrows():
        patient_id = row['patient_id']
        selection = row['selection']
        score = row['score']
        
        # 確定數據集類型
        if patient_id in train_ids:
            dataset_type = 'train'
        elif patient_id in val_ids:
            dataset_type = 'val'
        elif patient_id in test_ids:
            dataset_type = 'test'
        else:
            dataset_type = 'unknown'
            
        # 獲取類別
        action_type = get_action_type(selection)
        if action_type is None:
            continue
            
        # 判斷是否為正常人或病人
        if is_normal(score, patient_id):
            subject_type = 'Normal'
        elif is_patient(score, patient_id):
            subject_type = 'Patient'
        else:
            # 不在SCORE_THRESHOLDS定義範圍內，使用灰色
            color = '#808080'
            marker = 'o'
        
        class_name = f"{subject_type}-{action_type}"
        
        # 確定顏色和標記
        if class_name not in CLASS_CONFIG or CLASS_CONFIG[class_name] == 0:
            # 未啟用的類別使用黑色
            color = '#000000'
            marker = 'x'
        else:
            # 使用顏色映射中的顏色
            color = color_map.get(f"{class_name}_{patient_id}_{dataset_type}", '#808080')
            marker = 'o'
        
        if is_3d:
            ax.scatter(row[x_col], row[y_col], row[z_col], c=color, marker=marker)
        else:
            ax.scatter(row[x_col], row[y_col], c=color, marker=marker)
    
    # 設置標題和標籤
    dimension = "3D" if is_3d else "2D"
    plt.title(f'TSNE {dimension} Visualization (perplexity={perplexity})')
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    if is_3d:
        ax.set_zlabel('TSNE 3')
    
    # 保存圖片
    save_path = os.path.join(run_dir, f'tsne_{dimension.lower()}_p{perplexity}.png')
    plt.savefig(save_path)
    plt.close()

def visualize_tsne_results(run_dir: str, train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]):
    """
    視覺化TSNE結果
    
    Args:
        run_dir: 運行目錄
        train_ids: 訓練集病人ID集合
        val_ids: 驗證集病人ID集合
        test_ids: 測試集病人ID集合
    """
    # 讀取TSNE結果
    tsne_results = pd.read_csv('Model/WavFeatureFCNN/results/tsne_results.csv')
    
    # 生成2D和3D圖
    for perplexity in [5, 30, 50]:
        # 2D TSNE
        plot_tsne_visualization(tsne_results, run_dir, train_ids, val_ids, test_ids, perplexity, False)
        # 3D TSNE
        plot_tsne_visualization(tsne_results, run_dir, train_ids, val_ids, test_ids, perplexity, True)

def main():
    """主函數"""
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 獲取運行目錄
        run_dir = get_run_dir()
        logger.info(f"使用運行目錄: {run_dir}")
        
        # 保存配置信息
        save_config_info(run_dir)
        logger.info("配置信息已保存")
        
        # 啟用GPU記憶體增長
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("已啟用GPU記憶體增長")
        
        # 加載數據
        train_data, val_data, test_data, class_mapping, file_paths_list = load_data(CONFIG, logger)
        
        # 創建模型
        logger.info("構建模型...")
        model = WavFeatureCNN(CONFIG)
        
        # 編譯模型
        logger.info("編譯模型...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # 訓練模型
        logger.info("開始訓練...")
        history = train_model(model, train_data, val_data, CONFIG)
        
        # 保存訓練歷史
        history_path = os.path.join(run_dir, "training_history.json")
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)
        
        # 評估模型
        evaluate_model(model, test_data, class_mapping, logger, run_dir)
        
        # 生成TSNE可視化
        logger.info("開始生成TSNE可視化...")
        generate_tsne_plots()
        logger.info("TSNE可視化生成完成")
        
        # 保存模型
        model_save_path = os.path.join(run_dir, "model.keras")
        model.save(model_save_path)
        logger.info(f"模型已保存到: {model_save_path}")
        
    except Exception as e:
        logger.error(f"發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 