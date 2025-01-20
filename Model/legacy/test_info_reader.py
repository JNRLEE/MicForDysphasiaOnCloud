"""
測試讀取 info 文件的邏輯

這個腳本用於測試和調試從 Normal 和 Patient 目錄中讀取 info 文件的邏輯
"""

import os
import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

def setup_logging(log_dir: str) -> None:
    """設置日誌
    
    Args:
        log_dir: 日誌文件目錄
    """
    # 確保日誌目錄存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日誌文件名，包含時間戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'info_reader_{timestamp}.log')
    
    # 配置日誌格式
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # 同時輸出到控制台和文件
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logging.info(f"日誌文件位置: {log_file}")

@dataclass
class TestConfig:
    """測試配置"""
    selection_types: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.selection_types is None:
            self.selection_types = {
                'NoMovement': ['無動作', '無吞嚥'],
                'DrySwallow': ['乾吞嚥1口', '乾吞嚥2口', '乾吞嚥3口', '乾吞嚥'],
                'WaterDrinking': ['吞水10ml', '吞水20ml', '喝水', '吞水'],
                'Cracker': ['餅乾1塊', '餅乾2塊', '餅乾'],
                'Jelly': ['吞果凍', '果凍']
            }

class InfoReader:
    """Info 文件讀取器"""
    
    def __init__(self, wav_data_path: str):
        """初始化讀取器
        
        Args:
            wav_data_path: WavData 目錄的路徑
        """
        self.wav_data_path = wav_data_path
        self.config = TestConfig()
        
        # 驗證目錄結構
        self.normal_path = os.path.join(wav_data_path, 'NormalSubject')
        self.patient_path = os.path.join(wav_data_path, 'PatientSubject')
        
        if not os.path.exists(self.normal_path) or not os.path.exists(self.patient_path):
            raise ValueError(f"無法找到必要的目錄結構。請確保存在以下目錄：\n{self.normal_path}\n{self.patient_path}")
            
        logging.info(f"WavData 路徑: {wav_data_path}")
        logging.info(f"NormalSubject 路徑: {self.normal_path}")
        logging.info(f"PatientSubject 路徑: {self.patient_path}")
    
    def _load_info_file(self, subject_dir: str, exp_dir: str) -> Optional[Dict]:
        """加載info文件
        
        Args:
            subject_dir: 受試者目錄名（如 'N002' 或 'P001'）
            exp_dir: 實驗目錄名（如 'N002_2024-10-22_10-50-58' 或 'P001-1'）
            
        Returns:
            Optional[Dict]: info文件內容，如果找不到則返回None
        """
        # 根據受試者ID判斷是正常受試者還是病患受試者
        is_patient = subject_dir.startswith('P')
        subject_type = 'PatientSubject' if is_patient else 'NormalSubject'
        
        # 構建info文件的基礎路徑
        info_base = os.path.join(
            self.wav_data_path,
            subject_type,
            subject_dir,
            exp_dir
        )
        logging.debug(f"正在檢查目錄: {info_base}")
        
        # 構建可能的info文件路徑
        info_paths = [
            os.path.join(info_base, f"{subject_dir}_info"),
            os.path.join(info_base, f"{subject_dir}_info.json"),
            os.path.join(info_base, "info"),
            os.path.join(info_base, "info.json")
        ]
        
        for info_path in info_paths:
            if os.path.exists(info_path):
                logging.debug(f"找到info文件: {info_path}")
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logging.debug(f"文件內容:\n{content}")
                        
                        content = content.strip()
                        try:
                            # 首先嘗試作為 JSON 解析
                            info_dict = json.loads(content)
                            logging.debug(f"成功以 JSON 格式解析文件")
                        except json.JSONDecodeError as e:
                            logging.debug(f"JSON 解析失敗: {str(e)}")
                            # 如果 JSON 解析失敗，嘗試作為純文本解析
                            info_dict = {}
                            for line in content.split('\n'):
                                if ':' in line:
                                    key, value = line.strip().split(':', 1)
                                    info_dict[key.strip()] = value.strip()
                            logging.debug(f"成功以純文本格式解析文件")
                        
                        logging.debug(f"解析結果: {info_dict}")
                    
                    # 檢查是否成功讀取到 selection
                    if 'selection' in info_dict:
                        logging.debug(f"成功讀取到 selection: {info_dict['selection']}")
                        return info_dict
                    else:
                        logging.warning(f"文件中沒有 selection 字段")
                except Exception as e:
                    logging.error(f"加載文件出錯: {str(e)}")
                    continue
        
        logging.error(f"找不到任何可用的info文件或文件中沒有selection字段")
        return None
    
    def get_action_type_from_selection(self, selection: str) -> Optional[str]:
        """從selection字符串判斷動作類型"""
        selection_lower = selection.lower()
        for action_type, keywords in self.config.selection_types.items():
            if any(keyword.lower() in selection_lower for keyword in keywords):
                return action_type
        return None
    
    def test_read_info(self, subject_dir: str, exp_dir: str):
        """測試讀取特定目錄的info文件
        
        Args:
            subject_dir: 受試者目錄名（如 'N002' 或 'P001'）
            exp_dir: 實驗目錄名（如 'N002_2024-10-22_10-50-58' 或 'P001-1'）
        """
        logging.info(f"\n開始測試讀取 {subject_dir}/{exp_dir} 的info文件")
        
        info_dict = self._load_info_file(subject_dir, exp_dir)
        if info_dict:
            selection = info_dict.get('selection', '')
            action_type = self.get_action_type_from_selection(selection)
            logging.info(f"讀取結果:")
            logging.info(f"  - selection: {selection}")
            logging.info(f"  - 識別的動作類型: {action_type}")
            logging.info(f"  - 完整info內容: {info_dict}")
        else:
            logging.error("讀取失敗")
    
    def scan_and_test_all(self):
        """掃描並測試所有受試者的info文件"""
        # 測試 Normal 受試者
        for subject in os.listdir(self.normal_path):
            if subject.startswith('N'):
                subject_path = os.path.join(self.normal_path, subject)
                if os.path.isdir(subject_path):
                    for exp_dir in os.listdir(subject_path):
                        if os.path.isdir(os.path.join(subject_path, exp_dir)):
                            self.test_read_info(subject, exp_dir)
        
        # 測試 Patient 受試者
        for subject in os.listdir(self.patient_path):
            if subject.startswith('P'):
                subject_path = os.path.join(self.patient_path, subject)
                if os.path.isdir(subject_path):
                    for exp_dir in os.listdir(subject_path):
                        if os.path.isdir(os.path.join(subject_path, exp_dir)):
                            self.test_read_info(subject, exp_dir)

def main():
    """主函數"""
    # 設置基礎路徑
    base_path = "/content/drive/MyDrive/MicforDysphagia"
    wav_data_path = os.path.join(base_path, "WavData")
    log_dir = os.path.join(base_path, "logs")
    
    try:
        # 設置日誌
        setup_logging(log_dir)
        
        # 創建讀取器
        reader = InfoReader(wav_data_path)
        
        # 執行全面掃描測試
        reader.scan_and_test_all()
        
        # 或者測試特定目錄
        # reader.test_read_info('N003', 'N003_2024-10-22_11-35-23')
        # reader.test_read_info('P001', 'P001-1')
        
    except Exception as e:
        logging.error(f"執行測試時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 