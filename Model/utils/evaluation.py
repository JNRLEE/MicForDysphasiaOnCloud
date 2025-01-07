# 此代碼實現了模型評估和可視化功能

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict
import logging
from pathlib import Path
import json

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_dir: str
) -> None:
    """繪製混淆矩陣
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        class_names: 類別名稱列表
        save_dir: 保存目錄
    """
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 創建圖形
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('混淆矩陣')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    
    # 調整標籤顯示
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # 調整圖形佈局
    plt.tight_layout()
    
    # 保存圖片
    save_path = Path(save_dir) / 'confusion_matrix.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def evaluate_model(
    model,
    test_data: Dict[str, np.ndarray],
    test_labels: np.ndarray,
    save_dir: str,
    logger: logging.Logger
) -> Dict:
    """評估模型並生成報告
    
    Args:
        model: 訓練好的模型
        test_data: 測試數據
        test_labels: 測試標籤
        save_dir: 結果保存目錄
        logger: 日誌記錄器
    
    Returns:
        Dict: 評估結果
    """
    from Model.base.class_config import get_active_classes
    
    # 獲取類別名稱
    class_names = get_active_classes()
    
    # 預測
    y_pred = model.predict(test_data['encoder_input'])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_labels, axis=1)
    
    # 繪製混淆矩陣
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, save_dir)
    
    # 生成分類報告
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=class_names,
        digits=3,
        output_dict=True
    )
    
    # 記錄結果
    logger.info("分類報告：\n%s", classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=class_names,
        digits=3
    ))
    
    # 保存評估結果
    save_path = Path(save_dir) / 'evaluation_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report 