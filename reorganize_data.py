"""
此程式用於重組資料夾結構，將子資料夾中的第二層資料夾移動到父資料夾，並統計檔案名稱衝突情況
"""

import os
import shutil
from pathlib import Path
import logging
from collections import defaultdict

def setup_logger():
    """設置日誌記錄器"""
    logger = logging.getLogger('reorganize')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class ConflictTracker:
    """追蹤並記錄檔案名稱衝突"""
    def __init__(self):
        self.conflicts = defaultdict(list)
        self.total_moves = 0
        self.total_conflicts = 0
    
    def add_conflict(self, original_name: str, new_name: str, source_path: str):
        """記錄一個衝突"""
        self.conflicts[original_name].append({
            'new_name': new_name,
            'source': source_path
        })
        self.total_conflicts += 1
    
    def increment_moves(self):
        """增加移動計數"""
        self.total_moves += 1
    
    def print_report(self, logger: logging.Logger):
        """輸出衝突報告"""
        logger.info(f"\n=== 移動統計報告 ===")
        logger.info(f"總共移動的資料夾數量: {self.total_moves}")
        logger.info(f"發生名稱衝突的數量: {self.total_conflicts}")
        
        if self.conflicts:
            logger.info("\n=== 名稱衝突詳細資訊 ===")
            for original_name, conflicts in self.conflicts.items():
                logger.info(f"\n原始名稱: {original_name}")
                for conflict in conflicts:
                    logger.info(f"  - 重命名為: {conflict['new_name']}")
                    logger.info(f"    來源路徑: {conflict['source']}")

def safe_move_folder(source: Path, target_parent: Path, tracker: ConflictTracker, logger: logging.Logger) -> bool:
    """
    安全地移動資料夾，處理名稱衝突
    
    Args:
        source: 來源資料夾路徑
        target_parent: 目標父資料夾路徑
        tracker: 衝突追蹤器
        logger: 日誌記錄器
    
    Returns:
        bool: 是否成功移動資料夾
    """
    try:
        target = target_parent / source.name
        original_name = source.name
        
        # 如果目標路徑已存在，重新命名
        if target.exists():
            new_name = f"{source.parent.name}_{source.name}"
            target = target_parent / new_name
            tracker.add_conflict(original_name, new_name, str(source))
            logger.info(f"資料夾已存在，重新命名為: {new_name}")
        
        # 移動資料夾
        shutil.move(str(source), str(target))
        tracker.increment_moves()
        
        # 確認移動成功
        if not target.exists():
            logger.error(f"移動失敗，目標資料夾不存在: {target}")
            return False
            
        if source.exists():
            logger.error(f"移動失敗，來源資料夾仍然存在: {source}")
            return False
            
        logger.info(f"成功移動資料夾: {source.name} -> {target.name}")
        return True
        
    except Exception as e:
        logger.error(f"移動資料夾時發生錯誤 {source} -> {target_parent}: {str(e)}")
        return False

def reorganize_folder(base_path: str, logger: logging.Logger) -> ConflictTracker:
    """
    重組資料夾結構
    
    Args:
        base_path: 要重組的基礎資料夾路徑
        logger: 日誌記錄器
    
    Returns:
        ConflictTracker: 衝突追蹤器
    """
    base_dir = Path(base_path)
    if not base_dir.exists():
        logger.error(f"資料夾不存在: {base_path}")
        return ConflictTracker()
    
    tracker = ConflictTracker()
    
    try:
        # 獲取所有子資料夾
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        logger.info(f"找到 {len(subdirs)} 個子資料夾")
        
        # 遍歷每個子資料夾
        for subdir in subdirs:
            logger.info(f"處理資料夾: {subdir}")
            
            # 獲取所有第二層資料夾
            second_level_dirs = [d for d in subdir.iterdir() if d.is_dir()]
            logger.info(f"找到 {len(second_level_dirs)} 個第二層資料夾")
            
            # 移動所有第二層資料夾到父資料夾
            move_success = True
            for dir_path in second_level_dirs:
                if not safe_move_folder(dir_path, base_dir, tracker, logger):
                    move_success = False
                    logger.error(f"移動資料夾失敗: {dir_path}")
                    break
            
            # 只有當所有資料夾都成功移動且資料夾為空時才刪除原資料夾
            if move_success and not any(subdir.iterdir()):
                try:
                    subdir.rmdir()  # 只刪除空資料夾
                    logger.info(f"成功刪除空資料夾: {subdir}")
                except Exception as e:
                    logger.error(f"刪除資料夾失敗 {subdir}: {str(e)}")
            else:
                logger.warning(f"資料夾 {subdir} 未被刪除，因為還有檔案或移動失敗")
        
        return tracker
        
    except Exception as e:
        logger.error(f"處理資料夾時發生錯誤: {str(e)}")
        return tracker

def main():
    logger = setup_logger()
    
    # 定義要處理的資料夾路徑
    autoencoder_path = "/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/WavData/Combined/AutoEncoderData"
    original_path = "/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/WavData/Combined/OriginalData"
    
    # 處理 AutoEncoderData
    logger.info("\n=== 處理 AutoEncoderData ===")
    autoencoder_tracker = reorganize_folder(autoencoder_path, logger)
    autoencoder_tracker.print_report(logger)
    
    # 處理 OriginalData
    logger.info("\n=== 處理 OriginalData ===")
    original_tracker = reorganize_folder(original_path, logger)
    original_tracker.print_report(logger)
    
    # 輸出總體統計
    logger.info("\n=== 總體統計 ===")
    total_moves = autoencoder_tracker.total_moves + original_tracker.total_moves
    total_conflicts = autoencoder_tracker.total_conflicts + original_tracker.total_conflicts
    logger.info(f"總共移動的資料夾數量: {total_moves}")
    logger.info(f"總共發生的名稱衝突數量: {total_conflicts}")

if __name__ == "__main__":
    main() 