"""
數據處理模組：負責數據收集和處理的核心類
主要功能：
1. AudioFileCollector: 文件收集與分類
   - 遍歷指定路徑收集音訊文件
   - 讀取JSON信息文件
   - 根據評分將數據分類為正常組和病人組(病人score>=3)

2. AudioDataset: 音訊數據處理
   - 批次處理WAV文件
   - 執行音訊特徵提取（對數功率譜）
   - 數據分割與標籤處理
   - 準備訓練和驗證數據集
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
import gc
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import soundfile as sf
import librosa

from .config import AudioConfig, DatasetConfig, TrainingConfig
from Function import utils

@dataclass
class AudioFileCollector:
    """音頻文件收集器"""
    config: DatasetConfig
    
    def read_json_file(self, file_path: str) -> Dict:
        """讀取JSON文件，支持多種編碼"""
        encodings = ['utf-8', 'latin1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    try:
                        # 嘗試 JSON 解析
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # 嘗試純文本解析
                        info_dict = {}
                        for line in content.split('\n'):
                            if ':' in line:
                                key, value = line.strip().split(':', 1)
                                info_dict[key.strip()] = value.strip()
                        return info_dict
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode file {file_path}")

    def get_action_type_from_selection(self, selection: str) -> Optional[str]:
        """從selection字符串判斷動作類型"""
        selection_lower = selection.lower()
        for eng_type, ch_types in self.config.selection_types.items():
            if any(ch_type.lower() in selection_lower for ch_type in ch_types):
                return eng_type
        return None

    def collect_files_with_score(self) -> Tuple[List[dict], List[dict]]:
        """收集文件並根據score分類"""
        normal_files = []  # score <= 3
        patient_files = []  # score > 3

        for base_path in self.config.base_paths:
            for root, _, files in os.walk(base_path):
                wav_files = [f for f in files if f.endswith('.wav') and not f.startswith('._')]
                info_files = [f for f in files if (f.endswith('_info') or f.endswith('_info.json'))
                             and not f.startswith('._')]

                if not (wav_files and info_files):
                    continue

                try:
                    info_path = os.path.join(root, info_files[0])
                    info_data = self.read_json_file(info_path)

                    score = info_data.get('score', 0)
                    selection = info_data.get('selection', '').strip()

                    if not selection:
                        continue

                    action_type = self.get_action_type_from_selection(selection)
                    if action_type is None or action_type not in self.config.action_types:
                        continue

                    wav_path = os.path.join(root, wav_files[0])
                    file_info = {
                        'path': wav_path,
                        'action_type': action_type,
                        'selection': selection,
                        'score': score
                    }

                    if score > self.config.score_threshold:
                        patient_files.append(file_info)
                    else:
                        normal_files.append(file_info)

                except Exception as e:
                    print(f"Error processing {root}: {e}")
                    continue

        return normal_files, patient_files

@dataclass
class AudioDataset:
    """音頻數據集"""
    wav_files: List[str]
    labels: List[int]
    audio_config: AudioConfig
    training_config: TrainingConfig

    def process_wav_files(self) -> Tuple[np.ndarray, np.ndarray]:
        """批次處理WAV文件，將音頻切分成固定長度的片段"""
        X_lps_list = []
        processed_labels = []

        print(f"\n開始處理音頻文件...")
        print(f"總文件數: {len(self.wav_files)}")
        print(f"批次大小: {self.audio_config.batch_proc_size}")

        for i in range(0, len(self.wav_files), self.audio_config.batch_proc_size):
            batch_files = self.wav_files[i:i + self.audio_config.batch_proc_size]
            batch_labels = self.labels[i:i + self.audio_config.batch_proc_size]
            
            print(f"\n處理批次 {i//self.audio_config.batch_proc_size + 1}:")
            print(f"文件列表: {[os.path.basename(f) for f in batch_files]}")

            try:
                # 加載和處理音頻文件
                X_wavs = utils.load_wavs(batch_files, self.audio_config.sr)
                if len(X_wavs) == 0:
                    print(f"警告：批次 {i//self.audio_config.batch_proc_size + 1} 沒有成功加載任何文件")
                    continue

                print(f"成功加載 {len(X_wavs)} 個音頻文件")
                print(f"音頻形狀: {X_wavs.shape}")

                # 提取對數功率譜特徵
                X_lps, _ = utils.lps_extract(
                    X_wavs, 
                    self.audio_config.frame_size,
                    self.audio_config.overlap,
                    self.audio_config.fft_size,
                    to_list=False
                )
                print(f"特徵形狀: {X_lps.shape}")

                # 轉置特徵
                X_lps = X_lps.transpose(0, 2, 1)  # (batch, time, freq)
                segments_per_file = []

                for j in range(len(X_lps)):
                    # 對每個文件進行分段
                    segments = utils.time_splite(
                        X_lps[j:j+1],  # 保持維度以便使用time_splite
                        time_len=self.audio_config.seq_num,
                        padding=False
                    )
                    if len(segments) > 0:
                        # 調整維度順序：(segments, 1, time_len) -> (segments, time_len)
                        segments = np.squeeze(segments, axis=1)
                        # 添加通道維度
                        segments = segments[..., np.newaxis]
                        X_lps_list.append(segments)
                        segments_per_file.append(len(segments))
                        # 處理標籤
                        processed_labels.extend([batch_labels[j]] * len(segments))

                print(f"成功處理批次，添加 {sum(segments_per_file)} 個分段")

                # 釋放內存
                del X_wavs
                gc.collect()

            except Exception as e:
                print(f"錯誤處理批次 {i//self.audio_config.batch_proc_size}: {str(e)}")
                print(f"詳細錯誤信息:", e)
                continue

        if not X_lps_list:
            raise ValueError("沒有成功處理任何音頻文件")

        # 合併所有處理後的數據
        X_combined = np.concatenate(X_lps_list, axis=0)
        y_combined = np.array(processed_labels)

        # 確保標籤和特徵數量匹配
        min_len = min(len(X_combined), len(y_combined))
        X_combined = X_combined[:min_len]
        y_combined = y_combined[:min_len]

        print(f"\n處理完成:")
        print(f"特徵形狀: {X_combined.shape}")
        print(f"標籤形狀: {y_combined.shape}")

        return X_combined, y_combined

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                  np.ndarray, np.ndarray, np.ndarray]:
        """準備訓練、驗證和測試數據"""
        # 先分割出測試集
        train_files, test_files, train_labels, test_labels = train_test_split(
            self.wav_files, self.labels,
            test_size=self.training_config.outside_test_ratio,
            random_state=42,
            stratify=self.labels
        )
        
        # 處理訓練數據
        train_dataset = AudioDataset(
            wav_files=train_files,
            labels=train_labels,
            audio_config=self.audio_config,
            training_config=self.training_config
        )
        X_train_val, y_train_val = train_dataset.process_wav_files()
        y_train_val_cat = to_categorical(y_train_val, num_classes=10)
        
        # 分割訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val_cat,
            test_size=self.training_config.validation_ratio,
            random_state=42,
            stratify=np.argmax(y_train_val_cat, axis=1)
        )
        
        # 處理測試數據
        test_dataset = AudioDataset(
            wav_files=test_files,
            labels=test_labels,
            audio_config=self.audio_config,
            training_config=self.training_config
        )
        X_test, y_test = test_dataset.process_wav_files()
        y_test_cat = to_categorical(y_test, num_classes=10)
        
        return X_train, X_val, X_test, y_train, y_val, y_test_cat

    def compute_class_weights(self, labels: np.ndarray):
        """計算類別權重"""
        # 段級別權重
        segment_labels = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
        segment_weights = compute_class_weight(
            'balanced',
            classes=np.unique(segment_labels),
            y=segment_labels
        )
        segment_weight_dict = dict(enumerate(segment_weights))
        
        # 文件級別權重
        file_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        file_weight_dict = dict(enumerate(file_weights))
        
        return {
            'segment_weights': segment_weight_dict,
            'file_weights': file_weight_dict
        }



