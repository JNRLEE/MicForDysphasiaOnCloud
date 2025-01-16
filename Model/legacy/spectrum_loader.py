"""
# 光譜特徵資料載入器
# 用於載入預處理後的光譜特徵數據，包括原始特徵和增強後的特徵
"""

import os
import json
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import yaml
from .class_config import get_class_mapping, get_active_classes

class SpectrumDataLoader:
    def __init__(self, data_dir: str, original_data_dir: str):
        """
        初始化光譜特徵資料載入器
        
        Args:
            data_dir: 預處理後的數據目錄路徑
            original_data_dir: 原始數據目錄路徑
        """
        self.data_dir = data_dir
        self.original_data_dir = original_data_dir
        
        if not os.path.exists(data_dir):
            raise ValueError(f"找不到預處理數據目錄: {data_dir}")
        if not os.path.exists(original_data_dir):
            raise ValueError(f"找不到原始數據目錄: {original_data_dir}")
            
        # 讀取配置文件
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "experiments/baseline/config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 設置默認參數
        self.target_dim = self.config.get('feature_processing', {}).get('target_dim', 129)
        self.frame_size = self.config.get('feature_processing', {}).get('frame_size', 256)
        self.overlap = self.config.get('feature_processing', {}).get('overlap', 128)
        self.fft_size = self.config.get('feature_processing', {}).get('fft_size', 256)
        
        logging.info(f"目標特徵維度: {self.target_dim}")
        logging.info(f"幀大小: {self.frame_size}")
        logging.info(f"重疊大小: {self.overlap}")
        logging.info(f"FFT大小: {self.fft_size}")

    def _get_experiment_dirs(self):
        """獲取所有實驗目錄的路徑"""
        experiment_dirs = []
        
        # 遍歷數據目錄
        for patient_dir in os.listdir(self.data_dir):
            if not patient_dir.startswith('P'):  # 只處理以 P 開頭的目錄
                continue
                
            patient_path = os.path.join(self.data_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
                
            # 遍歷實驗目錄
            for exp_dir in os.listdir(patient_path):
                exp_path = os.path.join(patient_path, exp_dir)
                if not os.path.isdir(exp_path):
                    continue
                    
                # 檢查是否存在必要的文件
                if not os.path.exists(os.path.join(exp_path, 'original_lps.npy')):
                    continue
                    
                experiment_dirs.append(exp_path)
                
        logging.info(f"找到 {len(experiment_dirs)} 個實驗目錄")
        return experiment_dirs

    def _get_label_from_selection(self, selection: str, patient_id: str) -> Optional[str]:
        """
        根據選擇和病人ID獲取標籤
        
        Args:
            selection: 選擇的動作
            patient_id: 病人ID
            
        Returns:
            標籤字符串
        """
        # 獲取配置中的選擇類型映射
        selection_types = self.config.get('dataset', {}).get('selection_types', {})
        
        # 判斷病人類型
        is_patient = not patient_id.startswith('N')  # 假設正常人ID以N開頭
        prefix = "Patient-" if is_patient else "Normal-"
        
        # 從實驗名稱中提取動作類型
        if "DrySwallow" in selection:
            return f"{prefix}DrySwallow"
        elif "H2O" in selection or "Water" in selection:
            return f"{prefix}WaterDrinking"
        elif "Cookie" in selection:
            return f"{prefix}Cracker"
        elif "Jelly" in selection:
            return f"{prefix}Jelly"
        elif "NoMovement" in selection or "無動作" in selection:
            return f"{prefix}NoMovement"
            
        logging.error(f"無法將選擇 '{selection}' 映射到任何動作類型")
        logging.error(f"可用的選擇類型:")
        for action_type, selections in selection_types.items():
            logging.error(f"  {action_type}: {selections}")
        return None

    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        """
        標準化特徵維度
        
        Args:
            features: 輸入特徵數組
            
        Returns:
            標準化後的特徵數組
        """
        current_dim = features.shape[1]
        
        if current_dim == self.target_dim:
            return features
            
        elif current_dim < self.target_dim:
            # 填充到目標維度
            pad_width = ((0, 0), (0, self.target_dim - current_dim))
            return np.pad(features, pad_width, mode='constant', constant_values=0)
            
        else:
            # 截斷到目標維度
            return features[:, :self.target_dim]

    def _get_original_info_path(self, processed_dir):
        """根據處理後的目錄路徑獲取原始 info 文件的路徑"""
        try:
            # 從處理後的目錄路徑中提取實驗目錄名
            exp_dir = os.path.basename(processed_dir)
            patient_id = exp_dir.split('_')[0]
            
            logging.info(f"正在處理病人 {patient_id} 的數據，實驗: {exp_dir}")
            
            # 構建原始數據目錄路徑
            original_exp_dir = os.path.join(self.original_data_dir, patient_id, exp_dir)
            if not os.path.exists(original_exp_dir):
                logging.error(f"找不到原始數據目錄: {original_exp_dir}")
                return None
            
            # 檢查 info 文件
            info_path = os.path.join(original_exp_dir, f"{patient_id}_info.json")
            if os.path.exists(info_path):
                logging.info(f"找到 info 文件: {info_path}")
                return info_path
            
            logging.error(f"在目錄中找不到 info 文件: {original_exp_dir}")
            return None
            
        except Exception as e:
            logging.error(f"獲取 info 文件路徑時出錯: {str(e)}")
            return None

    def _extract_features(self, processed_dir):
        """從處理後的目錄中提取特徵和標籤"""
        try:
            # 讀取特徵數據
            feature_path = os.path.join(processed_dir, 'original_lps.npy')
            if not os.path.exists(feature_path):
                logging.error(f"找不到特徵文件: {feature_path}")
                return None, None
            
            features = np.load(feature_path)
            features = self._standardize_features(features)
            
            # 獲取 info 文件路徑
            info_path = self._get_original_info_path(processed_dir)
            if info_path is None:
                return None, None
            
            # 讀取 info 文件
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"解析 info 文件時出錯: {e}")
                return None, None
            
            # 從 info 文件中提取標籤
            selection = info.get('selection', '')
            if not selection:
                # 如果 selection 為空，嘗試從目錄名中提取
                exp_dir = os.path.basename(processed_dir)
                if '乾吞嚥' in exp_dir:
                    selection = '乾吞嚥1口'
                elif '餅乾' in exp_dir:
                    selection = '餅乾1塊'
                elif '果凍' in exp_dir:
                    selection = '果凍1口'
                elif '吞水' in exp_dir:
                    selection = '吞水10ml'
                elif '無動作' in exp_dir:
                    selection = '無動作'
                else:
                    logging.error(f"無法從目錄名中提取選擇類型: {exp_dir}")
                    return None, None
                
            # 獲取病人ID
            patient_id = os.path.basename(processed_dir).split('_')[0]
            
            # 將選擇映射到標籤
            label = self._get_label_from_selection(selection, patient_id)
            if label is None:
                logging.error(f"無法將選擇 '{selection}' 映射到標籤")
                return None, None
            
            return features, label
            
        except Exception as e:
            logging.error(f"提取特徵時發生錯誤: {str(e)}")
            return None, None

    def _map_selection_to_action(self, selection):
        """將中文選擇映射到動作類型"""
        mapping = {
            '乾吞嚥1口': 'DrySwallow',
            '吞水10ml': 'WaterDrinking',
            '餅乾1塊': 'Cookie',
            '果凍1口': 'Jelly'
        }
        
        # 使用部分匹配來處理可能的變體
        for key in mapping:
            if key in selection:
                return mapping[key]
            
        logging.warning(f"未知的選擇類型: {selection}")
        return None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """載入所有數據"""
        all_features = []
        all_labels = []
        all_filenames = []
        all_patient_ids = []
        
        # 獲取類別映射
        class_mapping = get_class_mapping()
        active_classes = get_active_classes()
        
        # 遍歷所有實驗目錄
        for processed_dir in self._get_experiment_dirs():
            features, label = self._extract_features(processed_dir)
            
            if features is not None and label is not None:
                # 檢查標籤是否在活動類別中
                if label in active_classes:
                    all_features.append(features)
                    # 將標籤轉換為數字索引
                    label_idx = class_mapping[label]
                    all_labels.append(label_idx)
                    
                    # 添加文件名和病人ID
                    exp_name = os.path.basename(processed_dir)
                    all_filenames.append(exp_name)
                    patient_id = exp_name.split('_')[0]
                    all_patient_ids.append(patient_id)
                    
                    logging.info(f"成功處理: {exp_name} (受試者: {patient_id}, 類別: {label})")
        
        if not all_features:
            raise ValueError("沒有找到有效的數據")
            
        # 轉換為numpy數組
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        logging.info(f"載入了 {len(all_features)} 個樣本")
        logging.info(f"特徵形狀: {features_array.shape}")
        logging.info(f"標籤形狀: {labels_array.shape}")
        logging.info(f"文件名數量: {len(all_filenames)}")
        logging.info(f"病人ID數量: {len(set(all_patient_ids))}")
        
        return features_array, labels_array, all_filenames, all_patient_ids 