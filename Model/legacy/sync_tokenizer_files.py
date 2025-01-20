"""
此程式用於同步 AutoEncoderData 和 OriginalData 資料夾之間的 tokenizer 檔案
將 AutoEncoderData 中的 tokens 檔案移動到 OriginalData 對應資料夾中並重命名
"""

import os
from pathlib import Path
import shutil
import logging
import json
from typing import Dict, List, Tuple

def setup_logger() -> logging.Logger:
    """設置日誌記錄器"""
    logger = logging.getLogger('sync_tokenizer')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_folder_mapping(autoencoder_path: Path, original_path: Path, logger: logging.Logger) -> Dict[str, Tuple[Path, Path]]:
    """
    建立兩個資料夾之間的對應關係
    
    Args:
        autoencoder_path: AutoEncoderData 資料夾路徑
        original_path: OriginalData 資料夾路徑
        logger: 日誌記錄器
    
    Returns:
        Dict[str, Tuple[Path, Path]]: 資料夾名稱到兩邊路徑的對應字典
    """
    mapping = {}
    auto_folders = {d.name: d for d in autoencoder_path.iterdir() if d.is_dir()}
    orig_folders = {d.name: d for d in original_path.iterdir() if d.is_dir()}
    
    # 找出共同的資料夾名稱
    common_names = set(auto_folders.keys()) & set(orig_folders.keys())
    logger.info(f"找到 {len(common_names)} 個相符的資料夾")
    
    # 建立對應關係
    for name in common_names:
        mapping[name] = (auto_folders[name], orig_folders[name])
    
    return mapping

def find_token_files(folder: Path) -> Tuple[Path, Path]:
    """
    在資料夾中尋找 tokens 相關檔案
    
    Args:
        folder: 要搜尋的資料夾路徑
    
    Returns:
        Tuple[Path, Path]: (tokens_info.json 路徑, tokens.npy 路徑)
    """
    info_file = None
    npy_file = None
    
    for file in folder.glob("*"):
        if file.name.endswith("_tokens_info.json"):
            info_file = file
        elif file.name.endswith("_tokens.npy"):
            npy_file = file
    
    return info_file, npy_file

def safe_copy_file(source: Path, target: Path, logger: logging.Logger) -> bool:
    """
    安全地複製檔案
    
    Args:
        source: 來源檔案路徑
        target: 目標檔案路徑
        logger: 日誌記錄器
    
    Returns:
        bool: 是否成功複製
    """
    try:
        if not source.exists():
            logger.error(f"來源檔案不存在: {source}")
            return False
            
        # 如果目標檔案已存在，先備份
        if target.exists():
            backup_path = target.parent / f"{target.stem}_backup{target.suffix}"
            shutil.move(str(target), str(backup_path))
            logger.info(f"已備份現有檔案到: {backup_path}")
        
        # 複製檔案
        shutil.copy2(str(source), str(target))
        
        # 驗證複製是否成功
        if not target.exists():
            logger.error(f"複製失敗，目標檔案不存在: {target}")
            return False
            
        logger.info(f"成功複製檔案: {source.name} -> {target.name}")
        return True
        
    except Exception as e:
        logger.error(f"複製檔案時發生錯誤 {source} -> {target}: {str(e)}")
        return False

def sync_folders(mapping: Dict[str, Tuple[Path, Path]], logger: logging.Logger) -> Tuple[int, int]:
    """
    同步資料夾之間的檔案
    
    Args:
        mapping: 資料夾對應關係
        logger: 日誌記錄器
    
    Returns:
        Tuple[int, int]: (成功處理的資料夾數, 失敗的資料夾數)
    """
    success_count = 0
    failure_count = 0
    
    for folder_name, (auto_folder, orig_folder) in mapping.items():
        logger.info(f"\n處理資料夾: {folder_name}")
        
        # 尋找 tokens 檔案
        info_file, npy_file = find_token_files(auto_folder)
        
        if not info_file or not npy_file:
            logger.error(f"在 {auto_folder} 中找不到完整的 tokens 檔案")
            failure_count += 1
            continue
        
        # 設定目標檔案路徑
        target_info = orig_folder / "WavTokenizer_tokens_info.json"
        target_npy = orig_folder / "WavTokenizer_tokens.npy"
        
        # 複製檔案
        if safe_copy_file(info_file, target_info, logger) and \
           safe_copy_file(npy_file, target_npy, logger):
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count

def main():
    logger = setup_logger()
    
    # 定義資料夾路徑
    base_path = Path("/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/WavData/Combined")
    autoencoder_path = base_path / "AutoEncoderData"
    original_path = base_path / "OriginalData"
    
    # 檢查資料夾是否存在
    if not autoencoder_path.exists() or not original_path.exists():
        logger.error("找不到必要的資料夾")
        return
    
    # 建立資料夾對應關係
    mapping = get_folder_mapping(autoencoder_path, original_path, logger)
    
    # 同步檔案
    success_count, failure_count = sync_folders(mapping, logger)
    
    # 輸出統計資訊
    logger.info("\n=== 同步完成 ===")
    logger.info(f"成功處理的資料夾數量: {success_count}")
    logger.info(f"失敗的資料夾數量: {failure_count}")
    logger.info(f"總共處理的資料夾數量: {success_count + failure_count}")

if __name__ == "__main__":
    main() 