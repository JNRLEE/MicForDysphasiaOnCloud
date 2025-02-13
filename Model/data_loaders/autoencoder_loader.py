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

def _is_colab() -> bool:
    """檢查是否在 Google Colab 環境中運行"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

class AutoEncoderDataLoader:
    def __init__(self, data_dir: str, original_data_dir: str):
        # 初始化數據加載器
        self.data_dir = Path(data_dir)
        self.original_data_dir = Path(original_data_dir)
        self.logger = logging.getLogger(__name__)
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
        """標準化特徵形狀到目標形狀
        
        Args:
            features: 輸入特徵，形狀為 (batch_size, 512, feature_dim)
            target_shape: 目標形狀，形狀為 (512, feature_dim)
            
        Returns:
            標準化後的特徵，形狀為 (batch_size, 512, target_feature_dim)
        """
        # 獲取當前和目標維度
        batch_size = features.shape[0]
        _, target_features = target_shape
        
        # 檢查時間步長是否為512
        if features.shape[1] != 512 or target_shape[0] != 512:
            raise ValueError(f"時間步長必須為512: features={features.shape}, target={target_shape}")
        
        # 創建新的特徵數組
        standardized = np.zeros((batch_size, 512, target_features), dtype=features.dtype)
        
        # 對每個批次進行處理
        for i in range(batch_size):
            # 複製有效數據
            feat_size = min(features.shape[2], target_features)
            standardized[i, :, :feat_size] = features[i, :, :feat_size]
            
            # 記錄詳細的形狀信息
            if i == 0:  # 只記錄第一個樣本的信息，避免日誌過多
                self.logger.debug(f"Sample {i}: 原始形狀 {features[i].shape} -> 目標形狀 {(512, target_features)}")
        
        return standardized

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

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """加載數據並將特徵填充到相同長度

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str], List[str]]: 特徵數組、標籤數組、文件名列表和受試者ID列表
        """
        all_features = []
        all_labels = []
        all_filenames = []
        all_patient_ids = []
        max_length = 0
        
        # 第一次遍歷找出最大長度
        for exp_dir in self._find_experiment_dirs():
            try:
                label = self._get_label_from_directory(exp_dir)
                if label is None:
                    continue
                
                features, tokens, filename = self._process_experiment_dir(exp_dir)
                if features is None or len(features) == 0:
                    continue
                
                max_length = max(max_length, features.shape[2])
                
            except Exception as e:
                self.logger.error(f"處理實驗目錄失敗 {exp_dir}: {str(e)}")
                continue
        
        # 第二次遍歷，填充特到相同長度
        for exp_dir in self._find_experiment_dirs():
            try:
                label = self._get_label_from_directory(exp_dir)
                if label is None:
                    continue
                
                features, tokens, filename = self._process_experiment_dir(exp_dir)
                if features is None or len(features) == 0:
                    continue
                
                info_path = self._get_feature_info_path(exp_dir)
                if info_path and info_path.exists():
                    info_data = self._load_original_info(info_path)
                    if info_data:
                        patient_id = info_data.get('patientID', '')
                        if patient_id:
                            # 填充特徵到最大長度
                            if len(features.shape) == 3 and features.shape[1] == 512:
                                padded_features = np.pad(
                                    features,
                                    ((0, 0), (0, 0), (0,max_length - features.shape[2])),
                                    mode='constant',
                                    constant_values=0
                                )
                                all_features.append(padded_features)
                                all_labels.extend([label] * len(features))
                                all_filenames.extend([filename] * len(features))
                                all_patient_ids.extend([patient_id] * len(features))
                            else:
                                self.logger.warning(f"特徵形狀不正確: {features.shape}")
            
            except Exception as e:
                self.logger.error(f"處理實驗目錄失敗 {exp_dir}: {str(e)}")
                continue

        if len(all_features) == 0:
            raise ValueError("沒有找到有效的數據")

        # 將所有特徵堆疊成一個數組
        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.array(all_labels)
        
        self.logger.info(f"成功加載 {len(features_array)} 個特徵，形狀為 {features_array.shape}")
        self.logger.info(f"特徵已填充到最大長度: {max_length}")
        
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
        """合併特徵和標記數據
        
        Args:
            features: shape (batch, 512, feature_dim)
            tokens: shape (batch, 512, feature_dim) 或 (batch, 1, feature_dim)
            
        Returns:
            combined: shape (batch, 512, feature_dim * 2)
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
        self.logger.info(f"合併後的特徵形狀: {combined.shape}")
        
        return combined