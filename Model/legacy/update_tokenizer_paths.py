"""
此程式用於更新所有 WavTokenizer_tokens_info.json 檔案中的 original_file 路徑
將路徑改為相對路徑指向同資料夾下的 Probe0_RX_IN_TDM4CH0.wav
"""

import os
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple

def setup_logger() -> logging.Logger:
    """設置日誌記錄器"""
    logger = logging.getLogger('path_updater')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def update_json_file(file_path: Path, logger: logging.Logger) -> bool:
    """
    更新單個 json 檔案中的路徑
    
    Args:
        file_path: json 檔案路徑
        logger: 日誌記錄器
    
    Returns:
        bool: 是否成功更新
    """
    try:
        # 讀取 JSON 檔案
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新路徑
        data['original_file'] = './Probe0_RX_IN_TDM4CH0.wav'
        
        # 寫回檔案
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"成功更新檔案: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"更新檔案失敗 {file_path}: {str(e)}")
        return False

def process_folders(base_path: Path, logger: logging.Logger) -> Tuple[int, int]:
    """
    處理所有資料夾中的 json 檔案
    
    Args:
        base_path: 基礎資料夾路徑
        logger: 日誌記錄器
    
    Returns:
        Tuple[int, int]: (成功更新的檔案數, 失敗的檔案數)
    """
    success_count = 0
    failure_count = 0
    
    # 遍歷所有子資料夾
    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue
            
        logger.info(f"\n處理資料夾: {folder.name}")
        
        # 尋找 WavTokenizer_tokens_info.json
        json_file = folder / "WavTokenizer_tokens_info.json"
        if not json_file.exists():
            logger.warning(f"找不到檔案: {json_file}")
            continue
        
        # 更新檔案
        if update_json_file(json_file, logger):
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count

def main():
    logger = setup_logger()
    
    # 定義資料夾路徑
    base_path = Path("/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/WavData/CombinedData")
    
    # 檢查資料夾是否存在
    if not base_path.exists():
        logger.error("找不到資料夾")
        return
    
    # 處理所有檔案
    success_count, failure_count = process_folders(base_path, logger)
    
    # 輸出統計資訊
    logger.info("\n=== 更新完成 ===")
    logger.info(f"成功更新的檔案數量: {success_count}")
    logger.info(f"失敗的檔案數量: {failure_count}")
    logger.info(f"總共處理的檔案數量: {success_count + failure_count}")

if __name__ == "__main__":
    main() 