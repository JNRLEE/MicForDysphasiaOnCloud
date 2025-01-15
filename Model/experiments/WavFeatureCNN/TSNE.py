# 此代碼用於將特徵數據使用t-SNE進行降維可視化
# 功能：
# 1. 加載並標準化特徵數據
# 2. 使用t-SNE進行3D降維
# 3. 生成可視化圖像

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
import yaml
import logging
from typing import Tuple, List, Dict
import json

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from Model.data_loaders.WavFeatureCNN_loader import AutoEncoderDataLoader
from Model.base.class_config import get_active_class_names
from Model.experiments.WavFeatureCNN.train import normalize_feature_dim, load_data, setup_logger

def setup_visualization_dir() -> Path:
    """設置可視化結果保存目錄"""
    save_dir = Path('visualization_results/TSNE_3D')
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def prepare_data_for_tsne(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """準備用於t-SNE的數據
    
    Args:
        features: 特徵數據，形狀為 [batch_size, window_size, feature_dim]
        labels: 標籤數據
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 處理後的特徵和標籤
    """
    # 重塑特徵為2D數組
    num_samples = features.shape[0]
    flattened_features = features.reshape(num_samples, -1)
    
    # 標準化特徵
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(flattened_features)
    
    return normalized_features, labels

def perform_tsne(features: np.ndarray, perplexity: float = 30.0, n_iter: int = 1000) -> np.ndarray:
    """執行t-SNE降維到3D
    
    Args:
        features: 特徵數據
        perplexity: t-SNE的困惑度參數
        n_iter: 迭代次數
        
    Returns:
        np.ndarray: 降維後的特徵
    """
    tsne = TSNE(
        n_components=3,  # 改為3維
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42
    )
    return tsne.fit_transform(features)

def plot_tsne_3d(
    tsne_results: np.ndarray,
    labels: np.ndarray,
    patient_ids: List[str],
    save_dir: Path,
    title: str = 't-SNE 3D Visualization',
    elevation: int = 30,
    azimuth: int = 45
):
    """繪製3D t-SNE結果
    
    Args:
        tsne_results: t-SNE降維後的特徵
        labels: 標籤
        patient_ids: 受試者ID列表
        save_dir: 保存目錄
        title: 圖表標題
        elevation: 仰角
        azimuth: 方位角
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 獲取類別名稱
    class_names = get_active_class_names()
    
    # 為每個類別創建散點圖並添加標籤
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            scatter = ax.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                tsne_results[mask, 2],
                label=class_name,
                alpha=0.6,
                s=100  # 增加點的大小
            )
            
            # 為該類別的每個點添加patient ID標籤
            for j, (x, y, z) in enumerate(tsne_results[mask]):
                patient_id = patient_ids[np.where(mask)[0][j]]
                ax.text(
                    x, y, z,
                    patient_id,
                    fontsize=8,
                    alpha=0.7
                )
    
    # 設置視角
    ax.view_init(elevation, azimuth)
    
    # 設置標題和標籤
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_zlabel('t-SNE 3', fontsize=12)
    
    # 添加圖例
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    # 調整布局以確保所有元素可見
    plt.tight_layout()
    
    # 保存不同角度的圖片
    for elev in [0, 30, 60]:
        for azim in [0, 45, 90, 135, 180, 225, 270, 315]:
            ax.view_init(elev, azim)
            save_path = save_dir / f"{title.lower().replace(' ', '_')}_elev{elev}_azim{azim}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def save_tsne_results(
    tsne_results: np.ndarray,
    labels: np.ndarray,
    patient_ids: List[str],
    save_dir: Path
):
    """保存t-SNE結果
    
    Args:
        tsne_results: t-SNE降維後的特徵
        labels: 標籤
        patient_ids: 受試者ID列表
        save_dir: 保存目錄
    """
    results_dict = {
        'tsne_coordinates': tsne_results.tolist(),
        'labels': labels.tolist(),
        'patient_ids': patient_ids,
        'class_names': get_active_class_names()
    }
    
    save_path = save_dir / 'tsne_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

def main():
    """主函數"""
    # 設置日誌記錄器
    logger = setup_logger()
    
    try:
        # 讀取配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加載數據
        logger.info("加載數據...")
        features, labels, filenames, patient_ids = load_data()
        
        # 標準化特徵維度
        logger.info("標準化特徵維度...")
        features = normalize_feature_dim(features, config)
        
        # 準備t-SNE數據
        logger.info("準備t-SNE數據...")
        normalized_features, labels = prepare_data_for_tsne(features, labels)
        
        # 設置保存目錄
        save_dir = setup_visualization_dir()
        
        # 使用不同的困惑度參數執行t-SNE
        perplexities = [5, 30, 50]
        for perplexity in perplexities:
            logger.info(f"執行t-SNE (perplexity={perplexity})...")
            tsne_results = perform_tsne(normalized_features, perplexity=perplexity)
            
            # 繪製和保存結果
            plot_title = f't-SNE 3D Visualization (perplexity={perplexity})'
            plot_tsne_3d(tsne_results, labels, patient_ids, save_dir, title=plot_title)
            
            # 保存最後一次的t-SNE結果
            if perplexity == perplexities[-1]:
                save_tsne_results(tsne_results, labels, patient_ids, save_dir)
        
        logger.info(f"可視化結果已保存到: {save_dir}")
        
    except Exception as e:
        logger.error(f"t-SNE可視化過程中出錯: {str(e)}")
        logger.error("詳細錯誤:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 