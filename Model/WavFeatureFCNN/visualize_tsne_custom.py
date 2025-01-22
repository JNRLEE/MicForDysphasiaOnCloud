"""
此程式用於根據 class_config.py 的配置生成自定義的 TSNE 視覺化圖
包含以下功能：
1. 根據 patient_id、score 和配置將數據點分類為 Normal 或 Patient
2. 根據 selection 字段和配置將數據點分類為不同的動作類型
3. 根據 CLASS_CONFIG 標記數據點為啟用或未啟用
4. 生成 2D 和 3D 的 TSNE 圖，每種圖都包含三種不同的 perplexity 值
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict
import matplotlib.font_manager as fm

# 添加專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

# 設置中文字體
if sys.platform.startswith('win'):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
elif sys.platform.startswith('darwin'):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False

from Model.base.class_config import (
    CLASS_CONFIG,
    SCORE_THRESHOLDS,
    SELECTION_TYPES,
    SUBJECT_SOURCE_CONFIG,
    get_action_type,
    subject_source,
    is_normal,
    is_patient
)

def setup_logger() -> logging.Logger:
    """設置日誌記錄器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_dataset_info(run_dir: str) -> Dict[str, Set[str]]:
    """
    從數據分布文件中讀取數據集信息
    
    Args:
        run_dir: 運行目錄路徑
        
    Returns:
        Dict[str, Set[str]]: 包含訓練集、驗證集和測試集的受試者ID集合
    """
    dataset_info = {
        'train': set(),
        'val': set(),
        'test': set()
    }
    
    distribution_path = os.path.join(run_dir, 'DataDistribution.md')
    if not os.path.exists(distribution_path):
        return dataset_info
        
    current_dataset = None
    with open(distribution_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '## 訓練集' in line:
                current_dataset = 'train'
            elif '## 驗證集' in line:
                current_dataset = 'val'
            elif '## 測試集' in line:
                current_dataset = 'test'
            elif '- ID列表:' in line and current_dataset:
                ids = line.split(':')[1].strip().split(', ')
                dataset_info[current_dataset].update(ids)
                
    return dataset_info

def generate_color_palette(num_classes: int) -> Dict[str, str]:
    """
    生成顏色映射字典
    
    Args:
        num_classes: 需要的顏色數量
        
    Returns:
        Dict[str, str]: 顏色映射字典
    """
    # 基礎顏色系統
    base_colors = [
        '#FF0000',  # 紅
        '#00FF00',  # 綠
        '#0000FF',  # 藍
        '#FF00FF',  # 紫
        '#FFFF00',  # 黃
        '#00FFFF',  # 青
        '#FFA500',  # 橙
        '#800080',  # 深紫
        '#008000',  # 深綠
        '#800000'   # 深紅
    ]
    
    # 為每個類別生成三種深淺度的顏色
    color_map = {}
    active_classes = [cls for cls, active in CLASS_CONFIG.items() if active == 1]
    
    for idx, class_name in enumerate(active_classes):
        base_color = base_colors[idx % len(base_colors)]
        # 轉換為RGB
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        # 訓練集使用原始顏色
        color_map[f"{class_name}-train"] = base_color
        
        # 驗證集使用較暗的顏色
        darker_rgb = (int(r*0.7), int(g*0.7), int(b*0.7))
        color_map[f"{class_name}-val"] = f"#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}"
        
        # 測試集使用最暗的顏色
        darkest_rgb = (int(r*0.4), int(g*0.4), int(b*0.4))
        color_map[f"{class_name}-test"] = f"#{darkest_rgb[0]:02x}{darkest_rgb[1]:02x}{darkest_rgb[2]:02x}"
    
    # 添加特殊類別的顏色
    color_map["未啟用"] = "#808080"  # 灰色
    color_map["未定義"] = "#D3D3D3"  # 淺灰色
    
    return color_map

def classify_data_point(row: pd.Series, dataset_info: Dict[str, Set[str]]) -> str:
    """
    根據配置對數據點進行分類
    
    Args:
        row: 數據行
        dataset_info: 數據集信息
        
    Returns:
        str: 數據點的分類標籤
    """
    # 確定數據集類型
    if row['patient_id'] in dataset_info['train']:
        dataset_type = 'train'
    elif row['patient_id'] in dataset_info['val']:
        dataset_type = 'val'
    elif row['patient_id'] in dataset_info['test']:
        dataset_type = 'test'
    else:
        dataset_type = 'unknown'
    
    # 確定受試者類型和動作類型
    is_normal_subject, is_patient_subject = subject_source(row['score'], row['patient_id'])
    action_type = row['selection']  # 已經是英文類型
    
    if is_normal_subject:
        subject_type = 'Normal'
    elif is_patient_subject:
        subject_type = 'Patient'
    else:
        return "未定義"
    
    class_name = f"{subject_type}-{action_type}"
    
    # 檢查類別是否啟用
    if CLASS_CONFIG.get(class_name, 0) == 0:
        return "未啟用"
    
    return f"{class_name}-{dataset_type}"

def plot_tsne(tsne_results: pd.DataFrame,
             dataset_info: Dict[str, Set[str]],
             color_map: Dict[str, str],
             perplexity: int,
             is_3d: bool,
             save_path: str):
    """
    繪製TSNE視覺化圖
    
    Args:
        tsne_results: TSNE結果DataFrame
        dataset_info: 數據集信息
        color_map: 顏色映射字典
        perplexity: perplexity參數值
        is_3d: 是否為3D圖
        save_path: 保存路徑
    """
    plt.figure(figsize=(12, 8))
    if is_3d:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    
    # 為每個數據點添加分類標籤
    tsne_results['label'] = tsne_results.apply(
        lambda row: classify_data_point(row, dataset_info), axis=1
    )
    
    # 獲取座標列
    if is_3d:
        x_col = f'tsne_3d_p{perplexity}_x'
        y_col = f'tsne_3d_p{perplexity}_y'
        z_col = f'tsne_3d_p{perplexity}_z'
    else:
        x_col = f'tsne_2d_p{perplexity}_x'
        y_col = f'tsne_2d_p{perplexity}_y'
    
    # 定義繪圖順序
    plot_order = []
    
    # 1. 未啟用和未定義的數據點（最下層）
    special_labels = ["未啟用", "未定義"]
    plot_order.extend([label for label in special_labels if label in tsne_results['label'].unique()])
    
    # 2. 訓練集數據（中下層）
    train_labels = sorted([label for label in tsne_results['label'].unique() 
                         if label.endswith('-train') and label not in special_labels])
    plot_order.extend(train_labels)
    
    # 3. 驗證集數據（中上層）
    val_labels = sorted([label for label in tsne_results['label'].unique() 
                        if label.endswith('-val') and label not in special_labels])
    plot_order.extend(val_labels)
    
    # 4. 測試集數據（最上層）
    test_labels = sorted([label for label in tsne_results['label'].unique() 
                         if label.endswith('-test') and label not in special_labels])
    plot_order.extend(test_labels)
    
    # 按順序繪製散點圖
    for label in plot_order:
        mask = tsne_results['label'] == label
        color = color_map.get(label, '#000000')
        
        if is_3d:
            ax.scatter(
                tsne_results.loc[mask, x_col],
                tsne_results.loc[mask, y_col],
                tsne_results.loc[mask, z_col],
                c=color,
                label=label,
                alpha=0.7,
                s=50 if label in special_labels else 100  # 特殊類別的點稍小一些
            )
        else:
            ax.scatter(
                tsne_results.loc[mask, x_col],
                tsne_results.loc[mask, y_col],
                c=color,
                label=label,
                alpha=0.7,
                s=50 if label in special_labels else 100  # 特殊類別的點稍小一些
            )
    
    # 設置標題和標籤
    dimension = "3D" if is_3d else "2D"
    plt.title(f'TSNE {dimension} Visualization (perplexity={perplexity})')
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    if is_3d:
        ax.set_zlabel('TSNE 3')
    
    # 添加圖例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # 調整布局並保存
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def generate_tsne_plots(run_dir: str, logger: logging.Logger = None) -> None:
    """
    生成TSNE視覺化圖並保存到指定的運行目錄
    
    Args:
        run_dir: 運行目錄路徑
        logger: 日誌記錄器（可選）
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info("開始生成TSNE視覺化圖")
    
    try:
        # 讀取TSNE結果
        tsne_results_path = os.path.join(os.path.dirname(__file__), 'results', 'tsne_results.csv')
        if not os.path.exists(tsne_results_path):
            logger.error(f"找不到TSNE結果文件: {tsne_results_path}")
            return
            
        tsne_results = pd.read_csv(tsne_results_path)
        
        # 讀取數據集信息
        dataset_info = load_dataset_info(run_dir)
        
        # 生成顏色映射
        color_map = generate_color_palette(len([v for v in CLASS_CONFIG.values() if v == 1]))
        
        # 創建輸出目錄
        output_dir = os.path.join(run_dir, 'tsne_plots')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成不同參數組合的圖
        for perplexity in [5, 30, 50]:
            for is_3d in [False, True]:
                dimension = "3d" if is_3d else "2d"
                save_path = os.path.join(
                    output_dir,
                    f'tsne_{dimension}_p{perplexity}_custom.png'
                )
                
                logger.info(f"生成 {dimension.upper()} TSNE 圖 (perplexity={perplexity})")
                plot_tsne(
                    tsne_results,
                    dataset_info,
                    color_map,
                    perplexity,
                    is_3d,
                    save_path
                )
        
        logger.info("所有TSNE圖已生成完成")
        
    except Exception as e:
        logger.error(f"生成TSNE圖時發生錯誤: {str(e)}")
        raise

def main():
    """主函數"""
    logger = setup_logger()
    
    try:
        # 獲取最新的運行目錄
        run_dirs = sorted(glob.glob(os.path.join(project_root, 'runs', 'WavFeatureFCNN_*')))
        if not run_dirs:
            logger.error("找不到運行目錄")
            return
        latest_run_dir = run_dirs[-1]
        
        # 生成TSNE圖
        generate_tsne_plots(latest_run_dir, logger)
        
    except Exception as e:
        logger.error(f"主程序執行錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 