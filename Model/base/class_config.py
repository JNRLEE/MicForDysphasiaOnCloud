"""
類別配置：定義活動類別和相關功能
功能：
1. 定義類別配置字典和映射規則
2. 提供中文實驗類型到英文類別的映射
3. 提供數據過濾和標籤更新功能
4. 支持靈活的二分類和多分類配置

警告：此文件包含關鍵的類別配置和數據讀取邏輯，修改前必須諮詢負責人
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import os
import tensorflow as tf

# 評分閾值配置（可調整）
SCORE_THRESHOLDS = {
    'normal': 0,      # score <= 1 為正常人
    'patient': 10      # score >= 3 為病人
}

# 實驗類型映射字典（不可修改）
SELECTION_TYPES = {
    'NoMovement': ["無動作", "無吞嚥"],
    'DrySwallow': ["乾吞嚥1口", "乾吞嚥2口", "乾吞嚥3口", "乾吞嚥"],
    'Cracker': ["餅乾1塊", "餅乾2塊", "餅乾"],
    'Jelly': ["吞果凍", "果凍"],
    'WaterDrinking': ["吞水10ml", "吞水20ml", "喝水", "吞水"]
}

# 類別配置字典（通過設置1/0來控制分類）
CLASS_CONFIG = {
    'Normal-NoMovement': 1,
    'Normal-DrySwallow': 1,
    'Normal-Cracker': 0,
    'Normal-Jelly': 0,
    'Normal-WaterDrinking': 0,
    'Patient-NoMovement': 0,
    'Patient-DrySwallow': 0,
    'Patient-Cracker': 0,
    'Patient-Jelly': 0,
    'Patient-WaterDrinking': 0
}

def get_active_classes() -> List[str]:
    """獲取活動類別列表（為了向後兼容）
    
    Returns:
        List[str]: 活動類別列表
    """
    return [cls for cls, active in CLASS_CONFIG.items() if active == 1]

def get_active_class_names() -> List[str]:
    """獲取當前激活的類別名稱列表
    
    Returns:
        List[str]: 激活的類別名稱列表
    """
    return get_active_classes()  # 使用相同的實現以保持一致性

def get_num_classes() -> int:
    """獲取激活的類別數量
    
    Returns:
        int: 類別數量
    """
    return len(get_active_classes())

def get_class_mapping() -> Dict[str, int]:
    """獲取類別到索引的映射
    
    Returns:
        Dict[str, int]: 類別名稱到索引的映射字典
    """
    active_classes = get_active_classes()
    return {cls: idx for idx, cls in enumerate(active_classes)}

def validate_class_config() -> bool:
    """驗證類別配置是否有效
    
    Returns:
        bool: 配置是否有效
    """
    active_classes = sum(1 for v in CLASS_CONFIG.values() if v == 1)
    if active_classes < 2:
        raise ValueError("至少需要兩個活動的類別")
    return True

def is_class_active(class_name: str) -> bool:
    """檢查類別是否處於激活狀態
    
    Args:
        class_name: 類別名稱
        
    Returns:
        bool: 類別是否激活
    """
    return CLASS_CONFIG.get(class_name, 0) == 1

def convert_to_one_hot(labels: np.ndarray) -> np.ndarray:
    """將標籤轉換為 one-hot 編碼
    
    Args:
        labels: 原始標籤索引
        
    Returns:
        np.ndarray: one-hot 編碼後的標籤
    """
    num_classes = get_num_classes()
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)

def convert_from_one_hot(one_hot_labels: np.ndarray) -> np.ndarray:
    """將 one-hot 編碼轉換回標籤索引
    
    Args:
        one_hot_labels: one-hot 編碼的標籤
        
    Returns:
        np.ndarray: 標籤索引
    """
    return np.argmax(one_hot_labels, axis=1)

def update_labels(labels: np.ndarray, label_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """更新標籤以匹配活動類別
    
    Args:
        labels: 原始標籤
        label_names: 標籤名稱列表
        
    Returns:
        Tuple[np.ndarray, List[str]]: 更新後的標籤和活動類別列表
    """
    active_classes = get_active_classes()
    class_mapping = get_class_mapping()
    
    # 創建新標籤
    new_labels = np.array([class_mapping[label_names[i]] for i in labels])
    
    return new_labels, active_classes

def filter_data(data: np.ndarray, labels: np.ndarray, label_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """根據活動類別過濾數據
    
    Args:
        data: 特徵數據
        labels: 標籤
        label_names: 標籤名稱列表
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 過濾後的數據和標籤
    """
    active_classes = get_active_classes()
    
    # 創建掩碼
    mask = np.array([label_names[i] in active_classes for i in labels])
    
    # 過濾數據
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    
    return filtered_data, filtered_labels

def get_classification_mode() -> str:
    """獲取當前的分類模式
    
    Returns:
        str: 'binary' 或 'multi'
    """
    active_classes = sum(1 for v in CLASS_CONFIG.values() if v == 1)
    return 'binary' if active_classes == 2 else 'multi'

def is_patient(score: int) -> bool:
    """根據評分判斷是否為病人
    
    Args:
        score: 評分值
        
    Returns:
        bool: 是否為病人
    """
    return score >= SCORE_THRESHOLDS['patient']

def is_normal(score: int) -> bool:
    """根據評分判斷是否為正常人
    
    Args:
        score: 評分值
        
    Returns:
        bool: 是否為正常人
    """
    return score <= SCORE_THRESHOLDS['normal']

def get_action_type(selection: str) -> Optional[str]:
    """將中文實驗類型映射到英文類別
    
    Args:
        selection: 中文實驗類型名稱
        
    Returns:
        Optional[str]: 對應的英文類別名稱，如果沒有匹配則返回None
    """
    for action_type, cn_names in SELECTION_TYPES.items():
        if any(cn_name in selection for cn_name in cn_names):
            return action_type
    return None
    
def classify_sample(selection: str, score: int) -> Optional[str]:
    """根據實驗類型和評分確定完整的分類
    
    Args:
        selection: 中文實驗類型名稱
        score: 評分
        
    Returns:
        Optional[str]: 完整的分類名稱（例如：'Normal-NoMovement'）
    """
    action_type = get_action_type(selection)
    if action_type is None:
        return None
        
    # 根據評分確定是正常人還是病人
    if is_normal(score):
        subject_type = 'Normal'
    elif is_patient(score):
        subject_type = 'Patient'
    else:
        return None
        
    class_name = f"{subject_type}-{action_type}"
    
    # 檢查該類別是否在當前配置中激活
    if class_name not in CLASS_CONFIG or CLASS_CONFIG[class_name] == 0:
        return None
        
    return class_name

def read_info_json(info_path: str) -> Optional[Dict]:
    """讀取並解析info.json文件
    
    警告：此函數處理關鍵的實驗信息文件，不要隨意修改讀取邏輯，如需更改請先諮詢
    
    Args:
        info_path: info文件的路徑
        
    Returns:
        Optional[Dict]: 解析後的info數據，如果讀取失敗則返回None
    """
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading info file {info_path}: {str(e)}")
        return None

def get_class_from_info(info: Dict) -> Optional[str]:
    """從 info 字典中獲取類別名稱
    
    Args:
        info: info 字典，包含 selection 和 score 字段
        
    Returns:
        Optional[str]: 類別名稱，如果無法確定類別則返回 None
    """
    selection = info.get('selection')
    score = info.get('score')
    
    if selection is None or score is None:
        return None
        
    return classify_sample(selection, score) 