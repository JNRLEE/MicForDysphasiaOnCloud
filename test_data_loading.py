# 此腳本用於測試數據加載流程，包括讀取tokens_info.json和原始info.json文件

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

def read_json_file(file_path: str) -> Optional[Dict]:
    """讀取JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def get_original_info_path(tokens_info: Dict) -> Optional[str]:
    """從tokens_info中獲取原始info文件路徑"""
    original_file = tokens_info.get('original_file', '')
    if not original_file:
        return None
        
    # 將Google Drive路徑轉換為本地路徑
    local_path = original_file.replace(
        '/content/drive/MyDrive/MicforDysphagia/',
        '/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/'
    )
    
    # 獲取原始wav文件的目錄
    wav_dir = os.path.dirname(local_path)
    # 從wav文件名構造info文件名
    patient_id = os.path.basename(wav_dir).split('_')[0]
    info_path = os.path.join(wav_dir, f"{patient_id}_info.json")
    
    return info_path

def classify_data(score: int, selection: str) -> Optional[str]:
    """根據score和selection進行分類"""
    from Model.base.class_config import (
        SELECTION_TYPES,
        SCORE_THRESHOLDS,
        CLASS_CONFIG
    )
    
    # 確定是正常人還是病人
    if score == SCORE_THRESHOLDS['normal']:
        subject_type = 'Normal'
    elif score > SCORE_THRESHOLDS['patient']:
        subject_type = 'Patient'
    else:
        return None
        
    # 確定動作類型
    action_type = None
    for action, keywords in SELECTION_TYPES.items():
        if any(keyword in selection for keyword in keywords):
            action_type = action
            break
            
    if not action_type:
        return None
        
    # 構造完整的分類名稱
    class_name = f"{subject_type}-{action_type}"
    
    # 檢查該分類是否在CLASS_CONFIG中啟用
    if class_name not in CLASS_CONFIG or CLASS_CONFIG[class_name] == 0:
        return None
        
    return class_name

def process_single_directory(directory_path: str) -> None:
    """處理單個目錄"""
    # 查找tokens_info.json文件
    tokens_info_files = list(Path(directory_path).glob("*_tokens_info.json"))
    if not tokens_info_files:
        print(f"No tokens_info.json found in {directory_path}")
        return
        
    # 讀取tokens_info.json
    tokens_info = read_json_file(str(tokens_info_files[0]))
    if not tokens_info:
        return
        
    # 獲取並讀取原始info文件
    original_info_path = get_original_info_path(tokens_info)
    if not original_info_path:
        print(f"Could not determine original info path from {tokens_info_files[0]}")
        return
        
    original_info = read_json_file(original_info_path)
    if not original_info:
        return
        
    # 提取關鍵信息
    patient_id = original_info.get('patientID')
    score = original_info.get('score')
    selection = original_info.get('selection')
    
    if None in (patient_id, score, selection):
        print(f"Missing required fields in {original_info_path}")
        return
        
    # 進行分類
    class_name = classify_data(score, selection)
    
    print(f"\nProcessing directory: {directory_path}")
    print(f"Patient ID: {patient_id}")
    print(f"Score: {score}")
    print(f"Selection: {selection}")
    print(f"Classified as: {class_name}")

def main():
    """主函數"""
    # 測試一個具體的目錄
    test_dir = "WavData/AutoEncoderData/N004/N004_2024-10-22_13-38-48"
    process_single_directory(test_dir)

if __name__ == "__main__":
    main()
