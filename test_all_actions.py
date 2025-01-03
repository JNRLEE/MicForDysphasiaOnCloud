"""
測試所有動作類型的讀取

按照 data_structure.py 的結構：
1. 從 ProcessedData 讀取 .npy 文件
2. 根據路徑找到對應的 info 文件
3. 確保所有動作類型都能被正確讀取

標籤對應：
- 正常受試者(Normal): 0-4
  - NoMovement: 0
  - DrySwallow: 1
  - Cracker: 2
  - Jelly: 3
  - WaterDrinking: 4
- 病患受試者(Patient): 5-9
  - NoMovement: 5
  - DrySwallow: 6
  - Cracker: 7
  - Jelly: 8
  - WaterDrinking: 9
"""

import os
import json
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# 設置日誌格式
def setup_logging(log_dir: str) -> None:
    """設置日誌
    
    Args:
        log_dir: 日誌文件目錄
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'all_actions_test_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info(f"日誌文件位置: {log_file}")

@dataclass
class ActionConfig:
    """動作類型配置"""
    selection_types: Dict[str, List[str]] = None
    action_types: List[str] = None
    labels_str: List[str] = None
    
    def __post_init__(self):
        if self.selection_types is None:
            # 所有可能的中文描述
            self.selection_types = {
                'NoMovement': ['無動作', '無吞嚥'],
                'DrySwallow': ['乾吞嚥1口', '乾吞嚥2口', '乾吞嚥3口', '乾吞嚥'],
                'Cracker': ['餅乾1塊', '餅乾2塊', '餅乾'],
                'Jelly': ['吞果凍', '果凍'],
                'WaterDrinking': ['吞水10ml', '吞水20ml', '喝水', '吞水']
            }
        
        if self.action_types is None:
            # 所有動作類型
            self.action_types = [
                'NoMovement', 'DrySwallow', 'Cracker', 'Jelly', 'WaterDrinking'
            ]
        
        if self.labels_str is None:
            # 生成所有標籤
            self.labels_str = []
            for action in self.action_types:
                self.labels_str.append(f'Normal-{action}')
            for action in self.action_types:
                self.labels_str.append(f'Patient-{action}')

class DataTester:
    """數據測試器"""
    
    def __init__(self, base_path: str):
        """初始化測試器
        
        Args:
            base_path: 基礎路徑（WavData 目錄的路徑）
        """
        self.base_path = base_path
        self.config = ActionConfig()
        
        # 設置路徑
        self.processed_path = os.path.join(base_path, 'ProcessedData')
        self.normal_path = os.path.join(base_path, 'NormalSubject')
        self.patient_path = os.path.join(base_path, 'PatientSubject')
        
        # 驗證路徑
        for path in [self.processed_path, self.normal_path, self.patient_path]:
            if not os.path.exists(path):
                raise ValueError(f"路徑不存在: {path}")
        
        logging.info(f"基礎路徑: {base_path}")
        logging.info(f"處理數據路徑: {self.processed_path}")
        logging.info(f"正常受試者路徑: {self.normal_path}")
        logging.info(f"病患受試者路徑: {self.patient_path}")
        logging.info(f"動作類型: {self.config.action_types}")
        logging.info(f"標籤列表: {self.config.labels_str}")
    
    def _load_info_file(self, subject_dir: str, exp_dir: str) -> Optional[Dict]:
        """加載info文件
        
        Args:
            subject_dir: 受試者目錄名（如 'N002' 或 'P001'）
            exp_dir: 實驗目錄名（如 'N002_2024-10-22_10-50-58' 或 'P001-1'）
            
        Returns:
            Optional[Dict]: info文件內容
        """
        # 判斷受試者類型
        is_patient = subject_dir.startswith('P')
        subject_type = 'PatientSubject' if is_patient else 'NormalSubject'
        
        # 構建info文件路徑
        info_base = os.path.join(
            self.base_path,
            subject_type,
            subject_dir,
            exp_dir
        )
        logging.debug(f"檢查info文件目錄: {info_base}")
        
        # 可能的info文件名
        info_paths = [
            os.path.join(info_base, f"{subject_dir}_info"),
            os.path.join(info_base, f"{subject_dir}_info.json"),
            os.path.join(info_base, "info"),
            os.path.join(info_base, "info.json")
        ]
        
        # 嘗試讀取info文件
        for info_path in info_paths:
            if os.path.exists(info_path):
                logging.debug(f"找到info文件: {info_path}")
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logging.debug(f"文件內容:\n{content}")
                        
                        content = content.strip()
                        try:
                            # 嘗試JSON解析
                            info_dict = json.loads(content)
                            logging.debug("成功解析JSON格式")
                        except json.JSONDecodeError as e:
                            logging.debug(f"JSON解析失敗: {str(e)}")
                            # 嘗試純文本解析
                            info_dict = {}
                            for line in content.split('\n'):
                                if ':' in line:
                                    key, value = line.strip().split(':', 1)
                                    info_dict[key.strip()] = value.strip()
                            logging.debug("成功解析純文本格式")
                        
                        logging.debug(f"解析結果: {info_dict}")
                        
                        if 'selection' in info_dict:
                            return info_dict
                        else:
                            logging.warning("文件中沒有selection字段")
                except Exception as e:
                    logging.error(f"讀取文件出錯: {str(e)}")
                    continue
        
        logging.error(f"無法找到或讀取info文件")
        return None
    
    def get_action_type(self, selection: str) -> Optional[str]:
        """從selection字符串判斷動作類型"""
        selection_lower = selection.lower()
        for action_type, keywords in self.config.selection_types.items():
            if any(keyword.lower() in selection_lower for keyword in keywords):
                return action_type
        return None
    
    def get_label_index(self, subject_type: str, action_type: str) -> Optional[int]:
        """獲取標籤索引
        
        Args:
            subject_type: 'Normal' 或 'Patient'
            action_type: 動作類型
            
        Returns:
            Optional[int]: 標籤索引 (0-9)
        """
        label = f"{subject_type}-{action_type}"
        try:
            return self.config.labels_str.index(label)
        except ValueError:
            return None
    
    def process_npy_file(self, file_path: str) -> Tuple[Optional[str], Optional[int]]:
        """處理一個npy文件
        
        Args:
            file_path: npy文件路徑
            
        Returns:
            Tuple[Optional[str], Optional[int]]: (標籤名稱, 標籤索引)
        """
        # 從路徑提取信息
        path_parts = file_path.split(os.sep)
        try:
            processed_idx = path_parts.index('ProcessedData')
            subject_dir = path_parts[processed_idx + 1]  # N003
            exp_dir = path_parts[processed_idx + 2]      # N003_2024-10-22_11-36-49
            
            logging.debug(f"處理文件: {file_path}")
            logging.debug(f"受試者: {subject_dir}, 實驗: {exp_dir}")
            
            # 讀取info文件
            info_dict = self._load_info_file(subject_dir, exp_dir)
            if info_dict is None:
                return None, None
            
            # 獲取selection
            selection = info_dict.get('selection', '').strip()
            if not selection:
                logging.error("沒有selection值")
                return None, None
            
            # 識別動作類型
            action_type = self.get_action_type(selection)
            if action_type is None:
                logging.error(f"無法識別動作類型: {selection}")
                return None, None
            
            # 確定受試者類型
            subject_type = 'Normal' if subject_dir.startswith('N') else 'Patient'
            
            # 獲取標籤
            label = f"{subject_type}-{action_type}"
            label_idx = self.get_label_index(subject_type, action_type)
            
            logging.info(f"文件 {os.path.basename(file_path)}:")
            logging.info(f"  - 受試者類型: {subject_type}")
            logging.info(f"  - Selection: {selection}")
            logging.info(f"  - 動作類型: {action_type}")
            logging.info(f"  - 標籤: {label} (索引: {label_idx})")
            
            return label, label_idx
            
        except (ValueError, IndexError) as e:
            logging.error(f"處理文件路徑時出錯: {str(e)}")
            return None, None
    
    def test_all_files(self):
        """測試所有文件"""
        # 用於統計每個類別的樣本數
        label_counts = {label: 0 for label in self.config.labels_str}
        
        # 遍歷所有npy文件
        for root, _, files in os.walk(self.processed_path):
            for file in files:
                if file == 'original_lps.npy':
                    file_path = os.path.join(root, file)
                    label, label_idx = self.process_npy_file(file_path)
                    if label is not None:
                        label_counts[label] += 1
        
        # 輸出統計結果
        logging.info("\n=== 統計結果 ===")
        for label, count in label_counts.items():
            logging.info(f"{label}: {count} 個樣本")

def main():
    """主函數"""
    # 設置基礎路徑
    base_path = "/content/drive/MyDrive/MicforDysphagia/WavData"  # Colab路徑
    log_dir = "/content/drive/MyDrive/MicforDysphagia/logs"      # Colab日誌路徑
    
    try:
        # 設置日誌
        setup_logging(log_dir)
        
        # 創建測試器
        tester = DataTester(base_path)
        
        # 執行測試
        tester.test_all_files()
        
    except Exception as e:
        logging.error(f"執行測試時發生錯誤: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 