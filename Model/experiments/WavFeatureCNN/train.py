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
from collections import Counter
import yaml

# 設定 matplotlib 字型
import matplotlib
import matplotlib.font_manager as fm

# 檢查是否在 Colab 環境
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    # 在 Colab 中安裝中文字型
    try:
        import subprocess
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'fonts-noto-cjk'], check=True)
        
        # 重新載入字型快取
        fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        fm._load_fontmanager()
        
        # 設定預設字型
        matplotlib.rc('font', family='Noto Sans CJK JP')
    except Exception as e:
        print(f"警告：安裝字型時出錯: {str(e)}")
        matplotlib.rc('font', family='DejaVu Sans')
else:
    # 本地環境使用系統字型
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in ['Arial Unicode MS', 'Noto Sans CJK JP', 'Microsoft JhengHei']:
        if font in available_fonts:
            matplotlib.rc('font', family=font)
            break

# 確保可以顯示負號
matplotlib.rcParams['axes.unicode_minus'] = False

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
            raise ValueError(f"受試者數量（{n_subjects}）不足，無法保證每個個子集至少有一個受試者")
    
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
    
    # 根據受試者ID分割數據
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
    
    # 創建數據集
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    # 打印分割信息
    print_split_info("訓練", train_features, train_labels, [patient_ids[i] for i in train_indices])
    print_split_info("驗證", val_features, val_labels, [patient_ids[i] for i in val_indices])
    print_split_info("測試", test_features, test_labels, [patient_ids[i] for i in test_indices])
    
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

def kmeans_unify_features(
    raw_features: np.ndarray,
    n_clusters: int = 30
) -> np.ndarray:
    """
    此函式透過 k-means 將合併後的特徵資料統一處理為固定長度的表示。
    步驟：
    1. 將每個樣本的 shape 由 [1, 512, combined_dim] 轉換成 [512, combined_dim]。
    2. 對時間序列進行標準化。
    3. 使用 k-means 將其分成 n_clusters 群。
    4. 將每個樣本的 cluster label 統計成直方圖。
    
    Args:
        raw_features (np.ndarray): 合併後的特徵，shape=(batch_size, 512, combined_dim)
        n_clusters (int): k-means 的 cluster 數量
        
    Returns:
        np.ndarray: 統合後的二維特徵陣列，shape=(batch_size, n_clusters)
    """
    processed_list = []
    
    for i in range(len(raw_features)):
        sample = raw_features[i]  # shape=(512, combined_dim)
        
        # 標準化
        scaler = StandardScaler()
        normalized_sample = scaler.fit_transform(sample)
        
        # k-means 聚類
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(normalized_sample)
        labels = kmeans.labels_
        
        # 將 cluster label 轉為直方圖，取得 shape=(n_clusters,)
        histogram, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters))
        
        processed_list.append(histogram)
    
    # 將所有結果堆疊成 array
    output_array = np.array(processed_list)  # shape=(batch_size, n_clusters)
    return output_array

def create_model(input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
    """創建 CNN 分類器模型
    Args:
        input_shape: 輸入特徵的形狀
        num_classes: 分類類別數量
    Returns:
        編譯好的 Keras 模型
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    
    # 第一個卷積塊
    x = tf.keras.layers.Conv1D(64, 5, padding='same', name='conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu1')(x)
    x = tf.keras.layers.MaxPooling1D(2, name='pool1')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # 第二個卷積塊
    x = tf.keras.layers.Conv1D(128, 5, padding='same', name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Activation('relu', name='relu2')(x)
    x = tf.keras.layers.MaxPooling1D(2, name='pool2')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # 第三個卷積塊
    x = tf.keras.layers.Conv1D(256, 5, padding='same', name='conv3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.Activation('relu', name='relu3')(x)
    x = tf.keras.layers.MaxPooling1D(2, name='pool3')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # 全局平均池化
    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
    
    # 全連接層
    x = tf.keras.layers.Dense(256, activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense2')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier_output')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='cnn_classifier')
    
    # 使用 Adam 優化器，並設定較小的學習率
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

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
    confusion_mat = confusion_matrix(y_true, y_pred_classes)
    
    # 打印混淆矩陣（純文字格式）
    print('\n=== 混淆矩陣 ===')
    # 打印標題行
    print(f"{'':25}", end='')
    for name in class_names:
        print(f"{name:15}", end=' ')
    print('\n')
    
    # 打印每一行
    for i, (row, class_name) in enumerate(zip(confusion_mat, class_names)):
        print(f"{class_name:25}", end='')
        for val in row:
            print(f"{val:15}", end=' ')
        print()
    print('==================\n')

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
        features, labels, filenames, patient_ids = load_data()
        logger.info(f"加載了 {len(features)} 個樣本")
        logger.info(f"合併後的特徵形狀: {features.shape}")
        logger.info(f"標籤形狀: {labels.shape}")
        logger.info(f"文件名數量: {len(filenames)}")
        logger.info(f"受試者數量: {len(set(patient_ids))}")

        # 使用 k-means 壓縮特徵
        logger.info("開始進行特徵壓縮...")
        compressed_features = kmeans_unify_features(features, n_clusters=30)
        logger.info(f"壓縮後的特徵形狀: {compressed_features.shape}")
        logger.info("特徵壓縮完成")
        
        # 準備數據集
        (train_features, train_labels), (val_features, val_labels), (test_features, test_labels) = prepare_data(
            compressed_features,
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

        # 為 Conv1D 擴展維度
        train_features = np.expand_dims(train_features, axis=-1)  # shape=(samples, 30, 1)
        val_features = np.expand_dims(val_features, axis=-1)
        test_features = np.expand_dims(test_features, axis=-1)

        # 設置保存目錄
        save_dir = setup_save_dir()
        
        # 創建和編譯模型
        model = create_model(
            input_shape=(compressed_features.shape[1], 1),  # (30, 1)
            num_classes=num_classes
        )
        
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
        evaluate_model(model, test_data, save_dir, logger)
        
        # 保存訓練歷史
        save_history(history, save_dir)
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        logger.error(f"Traceback:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 