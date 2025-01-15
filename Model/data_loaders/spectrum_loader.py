"""
此代碼實現了頻譜特徵的數據加載器，支持從預處理後的數據中讀取 original_lps.npy 作為輸入特徵
並與 class_config.py 連動以實現動態分類控制


1. 原始數據結構 (Raw Data Structure)
/content/drive/MyDrive/MicforDysphagia/WavData/
├── OriginalData/
│   ├── N002/
│   │   └── N002_2024-10-22_10-50-58/
│   │       ├── N002_info.json            # 包含實驗信息（請讀取該資料中記錄的受試者資訊）
│   │       └── Probe0_RX_IN_TDM4CH0.wav  # 原始音頻文件
│   └── P001/
│       └── P001-1/
│           ├── P001_info.json
│           └── DrySwallow_FB_left_240527.wav
└── ... (其他受試者)

2. 預處理後的數據結構 (Processed Data Structure)
/content/drive/MyDrive/MicforDysphagia/WavData/
└── ProcessedData/
    ├── N002/
    │   └── N002_2024-10-22_10-50-58/
    │       ├── processing_info.json       # 預處理參數信息（請讀去該資料中記錄的npy結構）
    │       ├── original_lps.npy          # 原始特徵（請使用此npy作為訓練資料）
    │       ├── noise_augmented_lps.npy   # 噪聲增強特徵
    │       ├── shift_augmented_lps.npy   # 時移增強特徵
    │       └── mask_augmented_lps.npy    # 頻率遮罩增強特徵
    └── P001/
        └── P001-1/
            ├── processing_info.json
            └── ... (特徵文件，結構同上)


"""


from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import numpy as np
import logging
from pathlib import Path
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.base.class_config import (
    get_class_mapping,
    get_active_classes,
    get_num_classes,
    convert_to_one_hot,
    convert_from_one_hot,
    read_info_json,
    get_class_from_info,
    is_class_active
)

# 讀取配置文件
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../experiments/baseline/config.yaml')

def _is_colab() -> bool:
    """檢查是否在 Google Colab 環境中運行"""
    try:
        import google.colab
        return True
    except:
        return False

class SpectrumDataLoader:
    def __init__(self, data_dir: str, original_data_dir: str):
        """初始化數據加載器
        
        Args:
            data_dir: 預處理後的數據目錄，包含 original_lps.npy
            original_data_dir: 原始數據目錄，包含 info.json
        """
        self.data_dir = Path(data_dir)
        self.original_data_dir = Path(original_data_dir)
        self.logger = logging.getLogger(__name__)
        
        # 讀取配置文件
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.target_dim = self.config['feature_processing']['target_dim']
            self.frame_size = self.config['feature_processing']['frame_size']
            self.overlap = self.config['feature_processing']['overlap']
            self.fft_size = self.config['feature_processing']['fft_size']
        except Exception as e:
            self.logger.warning(f"無法讀取配置文件，使用默認特徵維度 129: {str(e)}")
            self.target_dim = 129
            self.frame_size = 256
            self.overlap = 128
            self.fft_size = 256

        self.is_colab = _is_colab()
        
        if not self.data_dir.exists():
            raise ValueError(f"預處理數據目錄不存在: {data_dir}")
        if not self.original_data_dir.exists():
            raise ValueError(f"原始數據目錄不存在: {original_data_dir}")

    def _find_experiment_dirs(self) -> List[Path]:
        """遞歸查找所有實驗目錄"""
        experiment_dirs = []
        
        # 遍歷所有受試者目錄
        self.logger.info(f"開始在 {self.data_dir} 中查找實驗目錄")
        for subject_dir in self.data_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            self.logger.info(f"處理受試者目錄: {subject_dir.name}")
            # 遍歷受試者下的所有子目錄
            for sub_dir in subject_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                    
                self.logger.info(f"處理子目錄: {sub_dir.name}")
                # 遍歷子目錄下的所有實驗目錄
                for exp_dir in sub_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue
                    self.logger.info(f"找到實驗目錄: {exp_dir}")
                    experiment_dirs.append(exp_dir)
            
        self.logger.info(f"總共找到 {len(experiment_dirs)} 個實驗目錄")
        return experiment_dirs

    def _convert_path(self, colab_path: str) -> str:
        """將路徑轉換為當前環境的正確路徑"""
        if not self.is_colab:
            return colab_path
            
        if not colab_path.startswith('/content/drive/MyDrive/'):
            path = colab_path.replace('/content/drive/My Drive/', '')
            path = path.replace('/content/drive/MyDrive/', '')
            return os.path.join('/content/drive/MyDrive', path)
            
        return colab_path

    def _get_original_info_path(self, processed_dir: Path) -> Optional[Path]:
        """從原始數據目錄中找到對應的 info.json 文件
        
        Args:
            processed_dir: 預處理數據目錄路徑
            
        Returns:
            Optional[Path]: info.json 文件路徑，如果找不到則返回 None
        """
        try:
            # 從預處理目錄路徑中提取信息
            relative_path = processed_dir.relative_to(self.data_dir)
            original_dir = self.original_data_dir / relative_path
            
            self.logger.info(f"尋找原始數據目錄: {original_dir}")
            
            # 檢查原始目錄是否存在
            if not original_dir.exists():
                self.logger.error(f"找不到原始數據目錄: {original_dir}")
                return None
            
            # 獲取受試者ID（從目錄路徑中提取）
            patient_id = processed_dir.parent.parent.name
            
            # 構建 info.json 文件名（格式：{patientID}_info.json）
            info_filename = f"{patient_id}_info.json"
            info_path = original_dir / info_filename
            
            self.logger.info(f"尋找 info 文件: {info_path}")
            
            if not info_path.exists():
                self.logger.error(f"找不到 info 文件: {info_path}")
                return None
            
            return info_path
            
        except Exception as e:
            self.logger.error(f"獲取原始 info.json 路徑失敗: {str(e)}")
            return None

    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        """標準化特徵維度"""
        if features is None:
            return None
        
        # 檢查並調整時間步長
        if len(features.shape) == 2:
            features = features.reshape(1, *features.shape)
        
        # 填充或截斷到目標維度
        batch_size, time_steps, feature_dim = features.shape
        
        if feature_dim > self.target_dim:
            start = (feature_dim - self.target_dim) // 2
            features = features[:, :, start:start + self.target_dim]
        else:
            pad_width = ((0, 0), (0, 0), (0, self.target_dim - feature_dim))
            features = np.pad(features, pad_width, mode='constant', constant_values=0)
        
        return features

    def _extract_features(self, feature_file: Path) -> Tuple[np.ndarray, str, str]:
        """從特徵文件中提取特徵、標籤和文件名"""
        try:
            self.logger.info(f"開始處理特徵文件: {feature_file}")
            # 加載 original_lps.npy
            features = np.load(str(feature_file))
            self.logger.info(f"成功加載特徵，形狀為: {features.shape}")
            
            # 獲取對應的 info.json 路徑（從原始數據目錄）
            info_path = self._get_original_info_path(feature_file.parent)
            if info_path is None:
                raise ValueError(f"找不到對應的 info.json 文件: {feature_file.parent}")
            
            # 加載 info.json 內容
            info_data = read_info_json(info_path)
            if info_data is None:
                self.logger.error(f"無法讀取 info.json 文件: {info_path}")
                raise ValueError(f"無法讀取 info.json 文件: {info_path}")
            
            # 獲取類別標籤
            label = get_class_from_info(info_data)
            if label is None:
                self.logger.error(f"無法從 info.json 獲取有效的類別標籤: {info_path}")
                raise ValueError(f"無法從 info.json 獲取有效的類別標籤: {info_path}")
            
            self.logger.info(f"獲取到類別標籤: {label}")
            
            # 加載 processing_info.json
            processing_info_path = feature_file.parent / "processing_info.json"
            if processing_info_path.exists():
                with open(processing_info_path, 'r', encoding='utf-8') as f:
                    processing_info = json.load(f)
                    self.logger.info("成功加載 processing_info.json")
            
            # 標準化特徵
            features = self._standardize_features(features)
            self.logger.info(f"標準化後的特徵形狀: {features.shape}")
            
            return features, label, feature_file.stem
            
        except Exception as e:
            self.logger.error(f"加載特徵文件失敗 {feature_file}: {str(e)}")
            raise

    def _get_label_from_directory(self, exp_dir: Path) -> Optional[str]:
        """從實驗目錄獲取標籤"""
        try:
            # 從原始數據目錄獲取 info.json
            info_path = self._get_original_info_path(exp_dir)
            if info_path is None:
                return None
                
            info_data = read_info_json(info_path)
            if info_data is None:
                return None
                
            return get_class_from_info(info_data)
            
        except Exception as e:
            self.logger.error(f"從目錄獲取標籤失敗 {exp_dir}: {str(e)}")
            return None

    def _process_experiment_dir(self, exp_dir: Path) -> Tuple[np.ndarray, str, str]:
        """處理單個實驗目錄
        
        Args:
            exp_dir: 實驗目錄路徑
            
        Returns:
            Tuple[np.ndarray, str, str]: (特徵數據, 標籤, 文件名)
        """
        try:
            # 尋找 original_lps.npy 文件
            lps_file = exp_dir / "original_lps.npy"
            if not lps_file.exists():
                raise ValueError(f"找不到 original_lps.npy 文件: {exp_dir}")
            
            return self._extract_features(lps_file)
            
        except Exception as e:
            raise ValueError(f"處理實驗目錄失敗 {exp_dir}: {str(e)}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """加載數據並返回特徵和標籤"""
        all_features = []
        all_labels = []
        all_filenames = []
        all_patient_ids = []
        
        # 獲取類別映射
        class_mapping = get_class_mapping()
        self.logger.info("類別映射:")
        for class_name, idx in class_mapping.items():
            self.logger.info(f"  {class_name}: {idx}")
        
        for exp_dir in self._find_experiment_dirs():
            try:
                # 獲取標籤
                label = self._get_label_from_directory(exp_dir)
                if label is None:
                    self.logger.warning(f"無法從目錄獲取標籤: {exp_dir}")
                    continue
                    
                if not is_class_active(label):
                    self.logger.info(f"跳過非活躍類別 {label}: {exp_dir}")
                    continue
                
                self.logger.info(f"處理目錄 {exp_dir}，類別: {label}")
                
                # 處理實驗數據
                features, label, filename = self._process_experiment_dir(exp_dir)
                if features is None or features.size == 0:
                    self.logger.warning(f"特徵為空: {exp_dir}")
                    continue
                
                # 轉換標籤為索引
                label_idx = class_mapping[label]
                
                # 獲取受試者ID
                patient_id = exp_dir.parent.parent.name
                
                all_features.append(features)
                all_labels.append(label_idx)
                all_filenames.append(filename)
                all_patient_ids.append(patient_id)
                
                self.logger.info(f"成功處理: {filename} (受試者: {patient_id}, 類別: {label})")
                
            except Exception as e:
                self.logger.error(f"處理實驗目錄失敗 {exp_dir}: {str(e)}")
                continue
        
        if not all_features:
            self.logger.error("沒有找到有效的數據")
            raise ValueError("沒有找到有效的數據")
        
        # 轉換為 numpy 數組
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        # 轉換為 one-hot 編碼
        labels_one_hot = convert_to_one_hot(labels_array)
        
        self.logger.info(f"成功加載 {len(features_array)} 個特徵，形狀為 {features_array.shape}")
        self.logger.info(f"標籤形狀: {labels_one_hot.shape}")
        
        return features_array, labels_one_hot, all_filenames, all_patient_ids