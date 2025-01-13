"""
此模組提供數據可視化工具
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class VisualizationTool:
    def __init__(self, save_dir: str):
        """初始化可視化工具
        
        Args:
            save_dir: 圖表保存目錄
        """
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(__name__)
        
        # 創建保存目錄
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 設置基本字型
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_history(self, history: Dict, save_name: str = 'training_history.png'):
        """繪製訓練歷史圖表
        
        Args:
            history: 訓練歷史字典
            save_name: 保存的文件名
        """
        try:
            plt.figure(figsize=(12, 4))
            
            # 繪製損失曲線
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # 繪製準確率曲線
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # 調整布局並保存
            plt.tight_layout()
            save_path = self.save_dir / save_name
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"已保存訓練歷史圖表到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"繪製訓練歷史圖表時出錯: {str(e)}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_name: str = 'confusion_matrix.png'
    ):
        """繪製混淆矩陣
        
        Args:
            confusion_matrix: 混淆矩陣數據
            class_names: 類別名稱列表
            save_name: 保存的文件名
        """
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            
            # 添加刻度標籤
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, class_names)
            
            # 添加數值標籤
            thresh = confusion_matrix.max() / 2.
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if confusion_matrix[i, j] > thresh else "black")
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # 調整布局並保存
            plt.tight_layout()
            save_path = self.save_dir / save_name
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"已保存混淆矩陣圖表到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"繪製混淆矩陣時出錯: {str(e)}")
    
    def plot_feature_distribution(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        save_name: str = 'feature_distribution.png'
    ):
        """繪製特徵分布圖
        
        Args:
            features: 特徵數據
            labels: 標籤數據
            class_names: 類別名稱列表
            save_name: 保存的文件名
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # 計算每個類別的特徵均值
            for i, class_name in enumerate(class_names):
                class_features = features[labels == i]
                mean_features = np.mean(class_features, axis=0)
                plt.plot(mean_features, label=class_name)
            
            plt.title('Feature Distribution by Class')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Mean Value')
            plt.legend()
            
            # 調整布局並保存
            plt.tight_layout()
            save_path = self.save_dir / save_name
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"已保存特徵分布圖到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"繪製特徵分布圖時出錯: {str(e)}") 