# 此代碼實現了自動編碼器的數據加載器，支持可變特徵形狀和統一的數據訪問

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import numpy as np
import logging
from pathlib import Path

from ..base.class_config import (
    SELECTION_TYPES,
    SCORE_THRESHOLDS,
    CLASS_CONFIG,
    get_action_type,
    is_normal,
    is_patient
)

class AutoEncoderDataLoader:
    def __init__(self, data_dir: str, original_data_dir: str):
        # 初始化數據加載器
        self.data_dir = Path(data_dir)
        self.original_data_dir = Path(original_data_dir)
        self.logger = logging.getLogger(__name__)
        
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
        """將Colab路徑轉換為本地路徑"""
        # 移除可能的前綴
        path = colab_path.replace('/content/drive/MyDrive/', '')
        path = path.replace('/content/drive/My Drive/', '')
        
        # 構建本地路徑
        local_path = os.path.join(
            '/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive',
            path
        )
        
        return local_path

    def _load_original_info(self, info_path: Path) -> Dict[str, Any]:
        """加載原始信息文件"""
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
            
            # 從wav文件名構造info文件名
            patient_id = wav_dir.name.split('_')[0]
            original_info_path = wav_dir / f"{patient_id}_info.json"
            
            # 讀取原始info文件
            if not original_info_path.exists():
                self.logger.error(f"找不到原始info文件: {original_info_path}")
                return {}
                
            with open(original_info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
                
            # 添加feature_shape信息
            info_data['feature_shape'] = tokens_info.get('feature_shape')
            return info_data
            
        except Exception as e:
            self.logger.error(f"加載信息文件失敗 {info_path}: {str(e)}")
            return {}

    def _standardize_features(self, features: np.ndarray, feature_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """標準化特徵形狀"""
        try:
            if feature_shape is None:
                return features

            # 確保特徵是2D數組
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # 如果特徵維度不匹配，進行調整
            if features.shape[1] != feature_shape[0]:
                if features.shape[1] < feature_shape[0]:
                    # 如果特徵維度小於目標維度，用0填充
                    pad_width = ((0, 0), (0, feature_shape[0] - features.shape[1]))
                    features = np.pad(features, pad_width, mode='constant')
                else:
                    # 如果特徵維度大於目標維度，截斷
                    features = features[:, :feature_shape[0]]

            return features

        except Exception as e:
            raise ValueError(f"標準化特徵失敗: {str(e)}")

    def _extract_features(self, feature_file: Path) -> Tuple[np.ndarray, str]:
        """從特徵文件中提取特徵和文件名"""
        try:
            # 加載npy文件
            data = np.load(str(feature_file), allow_pickle=True)
            
            # 如果data是object類型且沒有shape，嘗試獲取其內容
            if data.dtype == object and data.shape == ():
                data = data.item()
                
                # 如果是字典類型，嘗試獲取特徵數據
                if isinstance(data, dict):
                    if 'features' in data:
                        features = data['features']
                    elif 'tokens' in data:
                        features = data['tokens']
                    else:
                        raise ValueError(f"字典中找不到特徵數據，可用的鍵: {list(data.keys())}")
                elif isinstance(data, (list, np.ndarray)):
                    features = np.array(data)
                else:
                    raise ValueError(f"無法處理的數據類型: {type(data)}")
            else:
                features = data

            # 將特徵轉換為numpy數組並確保是float32類型
            features = np.array(features, dtype=np.float32)

            # 檢查特徵是否為空
            if features.size == 0:
                raise ValueError("特徵數據為空")

            # 檢查特徵的數據類型
            if not np.issubdtype(features.dtype, np.number):
                raise ValueError(f"特徵數據類型不是數值類型: {features.dtype}")

            # 標準化特徵形狀
            if len(features.shape) == 1:
                # 如果是1D數組，將其重塑為2D
                features = features.reshape(1, -1)
            elif len(features.shape) > 2:
                # 如果是3D或更高維度，將其展平為2D
                features = features.reshape(features.shape[0], -1)

            # 獲取文件名
            filename = feature_file.stem

            return features, filename

        except Exception as e:
            raise ValueError(f"加載特徵文件失敗 {feature_file}: {str(e)}")

    def load_data(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """加載數據
        
        Returns:
            Tuple[np.ndarray, List[str], List[str]]: 特徵數組、標籤列表和文件名列表
        """
        all_features = []
        all_labels = []
        all_filenames = []
        feature_shape = None
        
        # 遍歷每個實驗目錄
        for exp_dir in self._find_experiment_dirs():
            try:
                # 獲取標籤
                label = self._get_label_from_directory(exp_dir)
                if label is None:
                    continue
                    
                # 處理實驗目錄
                features, filenames = self._process_experiment_dir(exp_dir, feature_shape)
                
                # 如果是第一個成功加載的特徵，設置目標形狀
                if feature_shape is None and features.size > 0:
                    feature_shape = (features.shape[1],)
                    self.logger.info(f"設置目標特徵形狀為: {feature_shape}")
                
                # 檢查特徵形狀
                if len(all_features) > 0:
                    # 如果已經有特徵，檢查新特徵的形狀是否匹配
                    if features.shape[1:] != all_features[0].shape[1:]:
                        self.logger.warning(f"特徵形狀不匹配: 預期 {all_features[0].shape[1:]}, 實際 {features.shape[1:]}, 目錄: {exp_dir}")
                        continue
                
                # 添加到結果列表
                all_features.append(features)
                all_labels.extend([label] * len(features))
                all_filenames.extend(filenames)
                
            except Exception as e:
                self.logger.error(f"處理目錄失敗 {exp_dir}: {str(e)}")
                continue
                
        if not all_features:
            raise ValueError("沒有找到有效的數據")
            
        # 合併所有特徵
        try:
            features = np.concatenate(all_features, axis=0)
            
            # 確保特徵和標籤的數量匹配
            if len(features) != len(all_labels):
                raise ValueError(f"特徵數量 ({len(features)}) 與標籤數量 ({len(all_labels)}) 不匹配")
            if len(features) != len(all_filenames):
                raise ValueError(f"特徵數量 ({len(features)}) 與文件名數量 ({len(all_filenames)}) 不匹配")
            
            self.logger.info(f"成功加載 {len(features)} 個特徵，形狀為 {features.shape}")
            return features, all_labels, all_filenames
            
        except Exception as e:
            raise ValueError(f"合併特徵失敗: {str(e)}")

    def _process_experiment_dir(self, exp_dir: Path, feature_shape: Optional[Tuple[int, ...]] = None) -> Tuple[np.ndarray, List[str]]:
        """處理實驗目錄中的特徵文件"""
        try:
            # 查找所有特徵文件
            feature_files = list(exp_dir.glob("*_tokens.npy"))
            if not feature_files:
                raise ValueError(f"在目錄中找不到特徵文件: {exp_dir}")

            all_features = []
            all_filenames = []

            # 處理每個特徵文件
            for feature_file in feature_files:
                try:
                    features, filename = self._extract_features(feature_file)
                    features = self._standardize_features(features, feature_shape)
                    
                    all_features.append(features)
                    all_filenames.extend([filename] * len(features))
                except Exception as e:
                    self.logger.warning(f"處理特徵文件失敗 {feature_file}: {str(e)}")
                    continue

            if not all_features:
                raise ValueError("沒有成功加載任何特徵")

            # 合併所有特徵
            features = np.concatenate(all_features, axis=0)
            return features, all_filenames

        except Exception as e:
            raise ValueError(f"處理實驗目錄失敗 {exp_dir}: {str(e)}")
    def _get_label_from_directory(self, exp_dir: Path) -> Optional[str]:
        """從實驗目錄獲取標籤"""
        try:
            # 獲取特徵信息文件
            info_path = self._get_feature_info_path(exp_dir)
            if info_path is None:
                self.logger.warning(f"未找到tokens_info文件: {exp_dir}")
                return None
            
            # 加載原始info文件
            info_data = self._load_original_info(info_path)
            if not info_data:
                return None
            
            # 獲取分類信息
            score = float(info_data.get('score', -1))
            selection = info_data.get('selection', '')
            self.feature_shape = info_data.get('feature_shape')
            
            if None in (score, selection, self.feature_shape):
                self.logger.warning(f"缺少必要的信息字段: {exp_dir}")
                return None
            
            # 確定是正常人還是病人
            if is_normal(score):
                subject_type = 'Normal'
            elif is_patient(score):
                subject_type = 'Patient'
            else:
                return None
            
            # 確定動作類型
            action_type = get_action_type(selection)
            if not action_type:
                return None
            
            # 檢查該分類是否在CLASS_CONFIG中啟用
            class_name = f"{subject_type}-{action_type}"
            if class_name not in CLASS_CONFIG or CLASS_CONFIG[class_name] == 0:
                return None
            
            self.logger.info(f"成功處理 {exp_dir.name}: {class_name}")
            return class_name
            
        except Exception as e:
            self.logger.error(f"獲取標籤失敗 {exp_dir}: {str(e)}")
            return None
