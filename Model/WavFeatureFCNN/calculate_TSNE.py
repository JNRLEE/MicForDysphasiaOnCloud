"""
此代碼用於計算WavTokenizer特徵的t-SNE降維結果
包含2D和3D降維，每個維度使用3種不同的perplexity值
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import glob
import json
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 添加專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

# 直接在此定義實驗類型映射，不依賴 class_config.py
SELECTION_TYPES = {
    'NoMovement': ["無動作", "無吞嚥"],
    'DrySwallow': ["乾吞嚥1口", "乾吞嚥2口", "乾吞嚥3口", "乾吞嚥"],
    'Cracker': ["餅乾1塊", "餅乾2塊", "餅乾"],
    'Jelly': ["吞果凍", "果凍"],
    'WaterDrinking': ["吞水10ml", "吞水20ml", "喝水", "吞水"]
}

def get_selection_type(selection: str) -> str:
    """將中文實驗類型映射到英文類別"""
    for action_type, keywords in SELECTION_TYPES.items():
        if any(keyword in selection for keyword in keywords):
            return action_type
    return selection  # 如果找不到對應，返回原始值

def setup_logger() -> logging.Logger:
    """設置日誌記錄器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def calculate_tsne_variations(features: np.ndarray, logger: logging.Logger) -> Dict[str, np.ndarray]:
    """
    計算不同參數組合的t-SNE結果
    
    Args:
        features: 特徵矩陣
        logger: 日誌記錄器
    
    Returns:
        Dict[str, np.ndarray]: 不同參數組合的t-SNE結果
    """
    tsne_results = {}
    perplexities = [5, 30, 50]
    dimensions = [2, 3]
    
    # 標準化特徵
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    for dim in dimensions:
        for perp in perplexities:
            try:
                logger.info(f"計算 {dim}D t-SNE (perplexity={perp})")
                key = f'tsne_{dim}d_p{perp}'
                tsne = TSNE(
                    n_components=dim,
                    perplexity=perp,
                    random_state=42,
                    n_iter=1000,
                    init='pca'
                )
                result = tsne.fit_transform(features_scaled)
                tsne_results[key] = result
                logger.info(f"完成 {key} 計算")
            except Exception as e:
                logger.error(f"計算 t-SNE 時出錯 (dim={dim}, perp={perp}): {str(e)}")
                continue
    
    return tsne_results

def process_features() -> pd.DataFrame:
    """
    處理特徵文件並計算t-SNE
    
    Returns:
        pd.DataFrame: 包含所有結果的數據框
    """
    logger = setup_logger()
    data_rows = []
    all_features = []
    feature_metadata = []
    
    # 設置數據目錄
    data_dir = os.path.join(project_root, "WavData", "CombinedData")
    if not os.path.exists(data_dir):
        logger.error(f"找不到數據目錄: {data_dir}")
        return pd.DataFrame()
    
    # 查找所有特徵文件
    feature_files = glob.glob(os.path.join(data_dir, "**", "WavTokenizer_tokens.npy"), recursive=True)
    logger.info(f"找到 {len(feature_files)} 個特徵文件")
    
    # 第一次遍歷：確定最常見的特徵維度
    feature_shapes = []
    for feature_path in feature_files:
        try:
            data_dict = np.load(feature_path, allow_pickle=True).item()
            features = data_dict['features']
            if len(features.shape) == 3:
                feature_shapes.append(features.shape[1:])
        except Exception as e:
            logger.warning(f"無法加載特徵文件 {feature_path}: {str(e)}")
            continue
    
    if not feature_shapes:
        logger.error("沒有找到有效的特徵文件")
        return pd.DataFrame()
    
    # 找出最常見的維度
    time_dims = [shape[0] for shape in feature_shapes]
    feature_dims = [shape[1] for shape in feature_shapes]
    target_time = max(set(time_dims), key=time_dims.count)
    target_features = max(set(feature_dims), key=feature_dims.count)
    
    logger.info(f"目標維度: time={target_time}, features={target_features}")
    
    # 第二次遍歷：處理特徵
    for feature_path in tqdm(feature_files, desc="處理文件"):
        try:
            # 加載特徵
            data_dict = np.load(feature_path, allow_pickle=True).item()
            features = data_dict['features']
            
            # 檢查特徵維度
            if len(features.shape) != 3:
                logger.warning(f"特徵維度不正確: {feature_path}, shape={features.shape}")
                continue
            
            batch_size, time_steps, n_features = features.shape
            
            # 如果維度差異太大，跳過該文件
            if abs(time_steps - target_time) > target_time * 0.5 or \
               abs(n_features - target_features) > target_features * 0.5:
                logger.warning(f"維度差異過大，跳過文件: {feature_path}")
                continue
            
            # 處理每個樣本
            processed_features = []
            for i in range(batch_size):
                feat = features[i]
                
                # 調整時間維度
                if time_steps < target_time:
                    feat = np.pad(feat, ((0, target_time - time_steps), (0, 0)), mode='constant')
                elif time_steps > target_time:
                    feat = feat[:target_time, :]
                
                # 調整特徵維度
                if n_features < target_features:
                    feat = np.pad(feat, ((0, 0), (0, target_features - n_features)), mode='constant')
                elif n_features > target_features:
                    feat = feat[:, :target_features]
                
                processed_features.append(feat.flatten())
            
            # 獲取目錄信息
            dir_path = os.path.dirname(feature_path)
            rel_path = os.path.relpath(feature_path, os.path.join(project_root, "WavData"))
            
            # 加載病人信息
            patient_info_files = glob.glob(os.path.join(dir_path, "*_info.json"))
            patient_info_files = [f for f in patient_info_files if not f.endswith("tokens_info.json")]
            
            if not patient_info_files:
                logger.warning(f"找不到病人信息文件: {dir_path}")
                continue
            
            with open(patient_info_files[0], 'r', encoding='utf-8') as f:
                patient_info = json.load(f)
            
            # 提取信息，不進行任何過濾
            patient_id = patient_info.get('patientID', '')
            score = patient_info.get('score', -1)  # 使用-1表示未記錄的評分
            selection = patient_info.get('selection', '')
            selection_type = get_selection_type(selection)
            
            # 存儲特徵和元數據，不進行任何過濾
            all_features.extend(processed_features)
            for _ in range(len(processed_features)):
                feature_metadata.append({
                    'filepath': rel_path,
                    'patient_id': patient_id,
                    'score': score,
                    'selection': selection_type
                })
            
        except Exception as e:
            logger.error(f"處理文件時出錯 {feature_path}: {str(e)}")
            continue
    
    if not all_features:
        logger.error("沒有找到有效的特徵")
        return pd.DataFrame()
    
    try:
        # 合併所有特徵
        combined_features = np.vstack(all_features)
        logger.info(f"處理 {len(combined_features)} 個樣本")
        
        # 計算t-SNE
        tsne_results = calculate_tsne_variations(combined_features, logger)
        
        # 創建數據行
        for i, metadata in enumerate(feature_metadata):
            row_data = metadata.copy()
            for key, coords in tsne_results.items():
                for j, dim in enumerate(['x', 'y', 'z'][:coords.shape[1]]):
                    row_data[f'{key}_{dim}'] = coords[i][j]
            data_rows.append(row_data)
            
    except Exception as e:
        logger.error(f"計算t-SNE時出錯: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(data_rows)

def main():
    logger = setup_logger()
    logger.info("開始計算t-SNE")
    
    try:
        # 處理特徵
        df = process_features()
        
        if len(df) == 0:
            logger.error("沒有處理到任何數據，退出程序")
            return
        
        # 創建輸出目錄
        output_dir = os.path.join(project_root, "Model", "WavFeatureFCNN", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存結果
        output_path = os.path.join(output_dir, "tsne_results.csv")
        logger.info(f"保存結果到: {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"結果保存成功")
        
    except Exception as e:
        logger.error(f"主程序出錯: {str(e)}")
        raise

if __name__ == "__main__":
    main()