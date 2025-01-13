# 此代碼實現了自動編碼器的數據加載器，支持可變特徵形狀和統一的數據訪問
# 警告：此文件包含關鍵的數據讀取路徑和邏輯，修改前必須諮詢負責人

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import numpy as np
import logging
from pathlib import Path
from collections import Counter
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.base.class_config import (
    SELECTION_TYPES,
    SCORE_THRESHOLDS,
    CLASS_CONFIG,
    get_action_type,
    is_normal,
    is_patient,
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
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../experiments/WavFeatureCNN/config.yaml')

def _is_colab() -> bool:
    """檢查是否在 Google Colab 環境中運行"""
    try:
        import google.colab
        return True
    except:
        return False

class AutoEncoderDataLoader:
    def __init__(self, data_dir: str, original_data_dir: str):
        """初始化數據加載器
        
        Args:
            data_dir: 特徵數據目錄
            original_data_dir: 原始數據目錄
        """
        self.data_dir = Path(data_dir)
        self.original_data_dir = Path(original_data_dir)
        self.logger = logging.getLogger(__name__)
        
        # 讀取配置文件
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.target_dim = self.config['feature_processing']['target_dim']
        except Exception as e:
            self.logger.warning(f"無法讀取配置文件，使用默認特徵維度 2000: {str(e)}")
            self.target_dim = 2000

        self.is_colab = _is_colab()
        
        if not self.data_dir.exists():
            raise ValueError(f"特徵數據目錄不存在: {data_dir}")
        if not self.original_data_dir.exists():
            raise ValueError(f"原始數據目錄不存在: {original_data_dir}")

    def _find_experiment_dirs(self) -> List[Path]:
        """遞歸查找所有實驗目錄"""
        experiment_dirs = []
        
        # 遍歷所有受試者目錄
        for subject_dir in self.data_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            # 查找該受試者下的所有實驗目錄
            subject_exp_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
            self.logger.info(f"在 {subject_dir.name} 中找到 {len(subject_exp_dirs)} 個實驗目錄")
            experiment_dirs.extend(subject_exp_dirs)
            
        return experiment_dirs

    def _get_feature_info_path(self, directory: Path) -> Optional[Path]:
        """獲取特徵信息文件路徑"""
        info_files = list(directory.glob("*_tokens_info.json"))
        return info_files[0] if info_files else None

    def _convert_path(self, colab_path: str) -> str:
        """將路徑轉換為當前環境的正確路徑
        
        Args:
            colab_path: 原始路徑（通常是 Colab 格式）
            
        Returns:
            str: 轉換後的路徑
        """
        if not self.is_colab:
            # 在本地環境中，保持原始路徑不變
            return colab_path
            
        # 在 Colab 環境中，確保路徑格式正確
        if not colab_path.startswith('/content/drive/MyDrive/'):
            # 移除可能的前綴
            path = colab_path.replace('/content/drive/My Drive/', '')
            path = path.replace('/content/drive/MyDrive/', '')
            # 添加正確的 Colab 路徑前綴
            return os.path.join('/content/drive/MyDrive', path)
            
        return colab_path

    def _find_info_file(self, directory: Path) -> Optional[Path]:
        """在目錄中尋找 info 文件
        
        警告：此函數處理關鍵的實驗信息文件搜索邏輯，不要隨意修改，如需更改請先諮詢
        
        Args:
            directory: 要搜索的目錄
            
        Returns:
            Optional[Path]: info 文件路徑，如果找不到則返回 None
        """
        # 尋找所有以 _info.json 或 _info 結尾的文件
        info_files = list(directory.glob("*_info.json")) + list(directory.glob("*_info"))
        
        if not info_files:
            self.logger.error(f"在 {directory} 中找不到 info 文件")
            return None
            
        # 如果找到多個文件，使用第一個
        if len(info_files) > 1:
            self.logger.warning(f"在 {directory} 中找到多個 info 文件，使用第一個: {info_files[0]}")
            
        return info_files[0]

    def _load_original_info(self, info_path: Path) -> Dict[str, Any]:
        """加載原始信息文件
        警告：此函數處理關鍵的實驗信息文件，不要隨意修改讀取邏輯，如需更改請先諮詢
        """
        try:
            # 讀取tokens_info.json
            with open(info_path, 'r', encoding='utf-8') as f:
                tokens_info = json.load(f)
                
            # 獲取原始文件路徑
            original_file = tokens_info.get('original_file', '')
            if not original_file:
                self.logger.error(f"tokens_info中沒有original_file字段: {info_path}")
                return {}
                
            # 轉換路徑
            local_path = self._convert_path(original_file)
            wav_dir = Path(os.path.dirname(local_path))
            
            # 尋找 info 文件
            original_info_path = self._find_info_file(wav_dir)
            if original_info_path is None:
                return {}
                
            # 讀取原始info文件
            with open(original_info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
                
            # 添加feature_shape信息
            info_data['feature_shape'] = tokens_info.get('feature_shape')
            return info_data
            
        except Exception as e:
            self.logger.error(f"加載信息文件失敗 {info_path}: {str(e)}")
            return {}

    def _standardize_features(self, features: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """標準化特徵維度
        
        Args:
            features: 輸入特徵
            target_shape: 目標形狀
            
        Returns:
            標準化後的特徵
        """
        if features is None:
            return None
        
        # 檢查並調整時間步長
        if len(features.shape) == 2:
            features = features.reshape(1, *features.shape)
        
        # 填充或截斷到目標維度
        batch_size, time_steps, feature_dim = features.shape
        
        if feature_dim > self.target_dim:
            # 如果特徵維度超過目標維度，從中間截取
            start = (feature_dim - self.target_dim) // 2
            features = features[:, :, start:start + self.target_dim]
        else:
            # 如果特徵維度小於目標維度，用0填充
            pad_width = ((0, 0), (0, 0), (0, self.target_dim - feature_dim))
            features = np.pad(features, pad_width, mode='constant', constant_values=0)
        
        return features

    def _extract_features(self, feature_file: Path) -> Tuple[np.ndarray, np.ndarray, str]:
        """從特徵文件中提取特徵、標記和文件名
        
        Returns:
            Tuple[np.ndarray, np.ndarray, str]: (特徵數據, 標記數據, 文件名)
        """
        try:
            # 加載npy文件
            data = np.load(str(feature_file), allow_pickle=True)
            
            # 初始化特徵和標記
            features = None
            tokens = None
            
            # 如果data是object類型且沒有shape，嘗試獲取其內容
            if data.dtype == object and data.shape == ():
                data = data.item()
                
                # 如果是字典類型，分別獲取特徵和標記數據
                if isinstance(data, dict):
                    features = data.get('features')
                    tokens = data.get('tokens')
                    
                    if features is None and tokens is None:
                        raise ValueError(f"字典中找不到特徵或標記數據，可用的鍵: {list(data.keys())}")
                elif isinstance(data, (list, np.ndarray)):
                    # 如果是數組類型，假設它是特徵數據
                    features = np.array(data)
                else:
                    raise ValueError(f"無法處理的數據類型: {type(data)}")
            else:
                features = data

            # 獲取特徵信息文件
            info_path = feature_file.with_name(feature_file.stem + '_info.json')
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    feature_shape = info.get('feature_shape')
                    token_shape = info.get('token_shape')
                    
                    # 重塑特徵數據
                    if features is not None and feature_shape:
                        try:
                            features = np.array(features, dtype=np.float32).reshape(feature_shape)
                        except ValueError as e:
                            self.logger.warning(f"無法將特徵重塑為 {feature_shape}: {str(e)}")
                    
                    # 重塑標記數據
                    if tokens is not None and token_shape:
                        try:
                            tokens = np.array(tokens, dtype=np.float32).reshape(token_shape)
                            # 如果 tokens 的形狀是 (batch, 1, feature_dim)，擴展到 (batch, 512, feature_dim)
                            if tokens.shape[1] == 1:
                                tokens = np.repeat(tokens, 512, axis=1)
                        except ValueError as e:
                            self.logger.warning(f"無法將標記重塑為 {token_shape}: {str(e)}")

            # 確保數據類型正確
            if features is not None:
                features = np.array(features, dtype=np.float32)
            if tokens is not None:
                tokens = np.array(tokens, dtype=np.float32)

            # 如果有 tokens，合併 features 和 tokens
            if features is not None and tokens is not None:
                features = self._combine_features(features, tokens)

            # 記錄形狀信息
            self.logger.info(f"最終特徵形狀: {features.shape if features is not None else None}")

            # 獲取文件名
            filename = feature_file.stem

            return features, None, filename  # 返回合併後的特徵，tokens 設為 None

        except Exception as e:
            raise ValueError(f"加載特徵文件失敗 {feature_file}: {str(e)}")

    def _get_label_from_directory(self, exp_dir: Path) -> Optional[int]:
        """從目錄中獲取標籤索引
        
        Args:
            exp_dir: 實驗目錄路徑
            
        Returns:
            Optional[int]: 標籤索引，如果無法確定標籤則返回None
        """
        try:
            # 獲取特徵信息文件
            info_path = self._get_feature_info_path(exp_dir)
            if not info_path:
                self.logger.warning(f"找不到特徵信息文件: {exp_dir}")
                return None
                
            # 加載原始信息
            info_data = self._load_original_info(info_path)
            if not info_data:
                return None
                
            # 獲取評分和實驗類型
            score = info_data.get('score', -1)
            selection = info_data.get('selection', '')
            
            # 獲取動作類型
            action_type = get_action_type(selection)
            if not action_type:
                self.logger.warning(f"無法確定動作類型: {selection}")
                return None
                
            # 確定受試者類型
            if is_normal(score):
                subject_type = 'Normal'
            elif is_patient(score):
                subject_type = 'Patient'
            else:
                self.logger.warning(f"無法確定受試者類型，評分: {score}")
                return None
                
            # 構建完整的類別名稱
            class_name = f"{subject_type}-{action_type}"
            
            # 檢查該類別是否在當前配置中激活
            if class_name not in CLASS_CONFIG or CLASS_CONFIG[class_name] == 0:
                self.logger.info(f"類別 {class_name} 未激活")
                return None
                
            # 獲取類別映射
            class_mapping = get_class_mapping()
            if class_name not in class_mapping:
                self.logger.warning(f"類別 {class_name} 不在映射中")
                return None
                
            return class_mapping[class_name]
            
        except Exception as e:
            self.logger.error(f"獲取標籤失敗 {exp_dir}: {str(e)}")
            return None

    def load_data(self, return_tokens: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """加載數據並返回特徵和標籤
        
        Args:
            return_tokens: 此參數已棄用，保留是為了向後兼容
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str], List[str]]: (特徵數組, 標籤數組, 文件名列表, 受試者ID列表)
        """
        all_features = []
        all_labels = []
        all_filenames = []
        all_patient_ids = []
        
        # 遍歷所有實驗目錄
        for exp_dir in self._find_experiment_dirs():
            try:
                label = self._get_label_from_directory(exp_dir)
                if label is None:
                    continue
                
                features, _, filename = self._process_experiment_dir(exp_dir)
                if features is None or len(features) == 0:
                    continue
                    
                # 檢查特徵形狀
                if len(features.shape) != 3 or features.shape[1] != 512:
                    self.logger.warning(f"特徵形狀不正確: {features.shape}")
                    continue
                
                # 填充或截斷到固定長度2000
                batch_size, time_steps, feature_dim = features.shape
                desired_dim = 2000
                
                if feature_dim > desired_dim:
                    # 如果特徵維度超過2000，從中間截取
                    start = (feature_dim - desired_dim) // 2
                    features = features[:, :, start:start + desired_dim]
                else:
                    # 如果特徵維度小於2000，用0填充
                    pad_width = ((0, 0), (0, 0), (0, desired_dim - feature_dim))
                    features = np.pad(features, pad_width, mode='constant', constant_values=0)
                
                all_features.append(features)
                all_labels.extend([label] * len(features))
                all_filenames.extend([filename] * len(features))
                
                # 獲取 patient_id
                info_path = self._get_feature_info_path(exp_dir)
                if info_path:
                    info_data = self._load_original_info(info_path)
                    if info_data:
                        patient_id = info_data.get('patientID', '')
                        all_patient_ids.extend([patient_id] * len(features))
                else:
                    all_patient_ids.extend([''] * len(features))
                
            except Exception as e:
                self.logger.error(f"處理實驗目錄失敗 {exp_dir}: {str(e)}")
                continue
        
        if not all_features:
            raise ValueError("沒有找到有效的數據")
        
        # 合併所有特徵
        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.array(all_labels)
        
        self.logger.info(f"成功加載 {len(features_array)} 個特徵，形狀為 {features_array.shape}")
        
        return features_array, labels_array, all_filenames, all_patient_ids

    def _process_experiment_dir(self, exp_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """處理實驗目錄中的特徵文件
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: (特徵數據, 標記數據, 文件名列表)
        """
        try:
            # 查找所有特徵文件
            feature_files = list(exp_dir.glob("*_tokens.npy"))
            if not feature_files:
                raise ValueError(f"在目錄中找不到特徵文件: {exp_dir}")

            all_features = []
            all_tokens = []
            all_filenames = []

            # 處理每個特徵文件
            for feature_file in feature_files:
                try:
                    # 提取特徵和標記
                    features, tokens, filename = self._extract_features(feature_file)
                    
                    # 檢查特徵形狀
                    if features is not None:
                        if len(features.shape) != 3 or features.shape[1] != 512:
                            self.logger.warning(f"特徵形狀不正確: {features.shape}")
                            continue
                            
                        all_features.append(features)
                        all_filenames.extend([filename] * len(features))
                        
                        # 如果有標記數據，也添加到列表中
                        if tokens is not None:
                            all_tokens.append(tokens)
                        
                except Exception as e:
                    self.logger.warning(f"處理特徵文件失敗 {feature_file}: {str(e)}")
                    continue

            if not all_features:
                raise ValueError("沒有成功加載任何特徵")

            # 合併所有特徵
            features = np.concatenate(all_features, axis=0)
            
            # 如果有標記數據，合併它們
            tokens = np.concatenate(all_tokens, axis=0) if all_tokens else None
            
            return features, tokens, all_filenames

        except Exception as e:
            raise ValueError(f"處理實驗目錄失敗 {exp_dir}: {str(e)}")

    def _combine_features(self, features: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """合併特徵和標記數據，並填充到固定長度
        
        Args:
            features: shape (batch, 512, feature_dim)
            tokens: shape (batch, 512, token_dim) 或 (batch, 1, token_dim)
            
        Returns:
            combined: shape (batch, 512, target_dim)
        """
        if features is None or tokens is None:
            return features if features is not None else tokens
        
        # 確保 tokens 的時間步長與 features 相同
        if tokens.shape[1] == 1:
            tokens = np.repeat(tokens, features.shape[1], axis=1)
        
        # 檢查形狀是否匹配
        if features.shape[0] != tokens.shape[0] or features.shape[1] != tokens.shape[1]:
            raise ValueError(f"特徵和標記的形狀不匹配: features={features.shape}, tokens={tokens.shape}")
        
        # 在特徵維度上合併
        combined = np.concatenate([features, tokens], axis=-1)
        
        # 填充或截斷到固定長度
        batch_size, time_steps, feature_dim = combined.shape
        
        if feature_dim > self.target_dim:
            # 如果特徵維度超過目標維度，從中間截取
            start = (feature_dim - self.target_dim) // 2
            combined = combined[:, :, start:start + self.target_dim]
        else:
            # 如果特徵維度小於目標維度，用0填充
            pad_width = ((0, 0), (0, 0), (0, self.target_dim - feature_dim))
            combined = np.pad(combined, pad_width, mode='constant', constant_values=0)
        
        self.logger.info(f"合併後的特徵形狀: {combined.shape}")
        return combined