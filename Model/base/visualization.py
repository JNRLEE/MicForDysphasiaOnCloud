"""
可視化工具模塊：用於分析和可視化模型訓練結果
包括：
1. 嵌入空間可視化 (t-SNE, UMAP)
2. 訓練歷史可視化
3. 距離分布可視化
4. 混淆矩陣可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import confusion_matrix
import logging
from typing import List, Optional, Tuple, Dict
import os
import subprocess
import matplotlib
import matplotlib.font_manager as fm

class VisualizationTool:
    """可視化工具類"""
    
    def __init__(self, save_dir: str = 'visualization_results'):
        """初始化可視化工具
        
        Args:
            save_dir: 保存可視化結果的目錄
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 檢查是否在 Colab 環境
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
            
        if is_colab:
            # 在 Colab 中安裝中文字型
            try:
                subprocess.run(['apt-get', 'update'], check=True)
                subprocess.run(['apt-get', 'install', '-y', 'fonts-noto-cjk'], check=True)
                
                # 重新載入字型快取
                fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
                fm._load_fontmanager()
                
                # 設置 matplotlib 使用 Noto Sans CJK JP (包含繁體中文)
                matplotlib.rc('font', family='Noto Sans CJK JP')
            except Exception as e:
                logging.warning(f"安裝字型時出錯: {str(e)}")
                # 如果安裝失敗，嘗試使用系統預設字型
                matplotlib.rc('font', family='DejaVu Sans')
        else:
            # 本地環境使用系統字型
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            for font in ['Arial Unicode MS', 'Noto Sans CJK JP', 'Microsoft JhengHei']:
                if font in available_fonts:
                    matplotlib.rc('font', family=font)
                    break
            
        # 確保可以顯示負號
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 設定 seaborn 樣式
        sns.set_style("whitegrid")
        
    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: List[str],
        method: str = 'tsne',
        title: str = '嵌入空間可視化',
        filename: str = 'embeddings.png'
    ) -> None:
        """可視化嵌入空間
        
        Args:
            embeddings: 嵌入向量
            labels: 標籤
            label_names: 標籤名稱列表
            method: 降維方法，'tsne' 或 'umap'
            title: 圖表標題
            filename: 保存的文件名
        """
        # 降維
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # umap
            reducer = umap.UMAP(random_state=42)
            
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 繪圖
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6
        )
        
        # 添加圖例
        legend1 = plt.legend(
            scatter.legend_elements()[0],
            label_names,
            title="類別",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        plt.gca().add_artist(legend1)
        
        plt.title(f'{title} ({method.upper()})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"嵌入空間可視化已保存到: {filename}")
    
    def plot_training_history(
        self,
        history: Dict,
        metrics: Optional[List[str]] = None,
        filename: str = 'training_history.png'
    ) -> None:
        """可視化訓練歷史
        
        Args:
            history: 訓練歷史字典
            metrics: 要繪製的指標列表
            filename: 保存的文件名
        """
        if metrics is None:
            metrics = ['loss']
            
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
        if num_metrics == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            ax.plot(history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            
            ax.set_title(f'{metric} 曲線')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        logging.info(f"訓練歷史可視化已保存到: {filename}")
    
    def plot_distance_distribution(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        filename: str = 'distance_distribution.png'
    ) -> None:
        """可視化嵌入空間中的距離分布
        
        Args:
            embeddings: 嵌入向量
            labels: 標籤
            filename: 保存的文件名
        """
        # 計算所有樣本對之間的距離
        num_samples = len(embeddings)
        pos_distances = []
        neg_distances = []
        
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                distance = np.sum(np.square(embeddings[i] - embeddings[j]))
                if labels[i] == labels[j]:
                    pos_distances.append(distance)
                else:
                    neg_distances.append(distance)
        
        # 繪製分布圖
        plt.figure(figsize=(10, 6))
        plt.hist(
            pos_distances,
            bins=50,
            alpha=0.5,
            label='相同類別',
            density=True
        )
        plt.hist(
            neg_distances,
            bins=50,
            alpha=0.5,
            label='不同類別',
            density=True
        )
        
        plt.title('嵌入空間中的距離分布')
        plt.xlabel('歐氏距離')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        logging.info(f"距離分布可視化已保存到: {filename}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str],
        filename: str = 'confusion_matrix.png'
    ) -> None:
        """繪製混淆矩陣
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            label_names: 標籤名稱列表
            filename: 保存的文件名
        """
        # 計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        # 繪製熱力圖
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names
        )
        
        plt.title('混淆矩陣')
        plt.xlabel('預測類別')
        plt.ylabel('真實類別')
        
        # 旋轉標籤以避免重疊
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"混淆矩陣可視化已保存到: {filename}")
    
    def plot_embedding_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: List[str],
        method: str = 'tsne',
        plot_centroids: bool = True,
        filename: str = 'embedding_clusters.png'
    ) -> None:
        """可視化嵌入空間的聚類結果
        
        Args:
            embeddings: 嵌入向量
            labels: 標籤
            label_names: 標籤名稱列表
            method: 降維方法，'tsne' 或 'umap'
            plot_centroids: 是否繪製類別中心點
            filename: 保存的文件名
        """
        # 降維
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # umap
            reducer = umap.UMAP(random_state=42)
            
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 繪圖
        plt.figure(figsize=(12, 8))
        
        # 繪製散點圖
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6
        )
        
        # 繪製類別中心點
        if plot_centroids:
            centroids = []
            for i in range(len(label_names)):
                mask = labels == i
                if np.any(mask):
                    centroid = embeddings_2d[mask].mean(axis=0)
                    centroids.append(centroid)
                    plt.scatter(
                        centroid[0],
                        centroid[1],
                        c='black',
                        marker='*',
                        s=200,
                        label=f'中心點 {label_names[i]}'
                    )
        
        # 添加圖例
        legend1 = plt.legend(
            scatter.legend_elements()[0],
            label_names,
            title="類別",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        plt.gca().add_artist(legend1)
        
        if plot_centroids:
            plt.legend(title="中心點", loc="center left", bbox_to_anchor=(1, 0.2))
        
        plt.title(f'嵌入空間聚類可視化 ({method.upper()})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"聚類可視化已保存到: {filename}")
    
    def visualize_all(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: List[str],
        history: Optional[Dict] = None,
        prefix: str = ''
    ) -> None:
        """生成所有可視化結果
        
        Args:
            embeddings: 嵌入向量
            labels: 標籤
            label_names: 標籤名稱列表
            history: 訓練歷史字典
            prefix: 文件名前綴
        """
        # t-SNE可視化
        self.plot_embeddings(
            embeddings,
            labels,
            label_names,
            method='tsne',
            filename=f'{prefix}tsne_embeddings.png'
        )
        
        # UMAP可視化
        self.plot_embeddings(
            embeddings,
            labels,
            label_names,
            method='umap',
            filename=f'{prefix}umap_embeddings.png'
        )
        
        # 距離分布
        self.plot_distance_distribution(
            embeddings,
            labels,
            filename=f'{prefix}distance_distribution.png'
        )
        
        # 聚類可視化
        self.plot_embedding_clusters(
            embeddings,
            labels,
            label_names,
            method='tsne',
            filename=f'{prefix}embedding_clusters.png'
        )
        
        # 如果有訓練歷史，則繪製訓練曲線
        if history is not None:
            self.plot_training_history(
                history,
                metrics=['loss'],
                filename=f'{prefix}training_history.png'
            )
        
        logging.info("所有可視化結果已生成") 