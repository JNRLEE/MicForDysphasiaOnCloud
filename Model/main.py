"""
主程序入口
"""

import os
import sys
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np

# 添加項目根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Model.base.config import load_config
from Model.base.data_loader import TripletDataLoader
from Model.base.visualization import VisualizationTool
from Model.experiments.triplet_loss.model import TripletLossModel

def setup_logging():
    """配置日誌系統"""
    # 創建logs目錄（使用絕對路徑）
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日誌文件名（包含時間戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"日誌文件創建於: {log_file}")
    logging.info(f"Python 路徑: {sys.path}")
    logging.info(f"項目根目錄: {project_root}")
    
    return log_dir

def main():
    # 設置日誌
    log_dir = setup_logging()
    
    try:
        # 記錄開始時間
        start_time = datetime.now()
        logging.info("=== 訓練開始 ===")
        logging.info(f"開始時間: {start_time}")
        
        # 加載配置
        config = load_config(experiment_type='triplet_loss')
        logging.info("配置加載完成")
        
        # 創建保存目錄（使用絕對路徑）
        tensorboard_dir = os.path.join(log_dir, 'triplet_loss')
        embeddings_dir = os.path.join(project_root, 'Model', 'embeddings')
        viz_dir = os.path.join(project_root, 'Model', 'visualization_results')
        
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        logging.info("創建保存目錄完成")
        
        # 初始化數據加載器
        data_loader = TripletDataLoader(config)
        
        # 加載數據
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.load_data()
        logging.info(f"數據加載完成。訓練集: {x_train.shape}, 驗證集: {x_val.shape}, 測試集: {x_test.shape}")
        
        # 創建模型
        model = TripletLossModel(config.model)
        model.build()
        logging.info("模型構建完成")
        
        # 設置模型保存路徑
        model_save_path = os.path.join(project_root, config.training.model_save_path)
        
        # 設置回調函數
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.training.early_stopping_patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True
            ),
            TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1
            )
        ]
        
        # 訓練模型
        logging.info("開始訓練...")
        history = model.train(
            data_loader=data_loader,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            callbacks=callbacks
        )
        
        logging.info("訓練完成")
        
        # 保存模型
        model.save(model_save_path)
        logging.info(f"模型已保存到: {model_save_path}")
        
        # 生成嵌入向量
        train_embeddings = model.encode(x_train)
        test_embeddings = model.encode(x_test)
        
        # 保存嵌入向量（使用絕對路徑）
        np.save(os.path.join(embeddings_dir, 'train_embeddings.npy'), train_embeddings)
        np.save(os.path.join(embeddings_dir, 'test_embeddings.npy'), test_embeddings)
        np.save(os.path.join(embeddings_dir, 'train_labels.npy'), y_train)
        np.save(os.path.join(embeddings_dir, 'test_labels.npy'), y_test)
        
        logging.info("嵌入向量已保存")
        
        # 初始化可視化工具
        viz_tool = VisualizationTool(save_dir=viz_dir)
        
        # 生成訓練集的可視化結果
        logging.info("生成訓練集可視化結果...")
        viz_tool.visualize_all(
            embeddings=train_embeddings,
            labels=y_train,
            label_names=config.dataset.labels_str,
            history=history.history,
            prefix='train_'
        )
        
        # 生成測試集的可視化結果
        logging.info("生成測試集可視化結果...")
        viz_tool.visualize_all(
            embeddings=test_embeddings,
            labels=y_test,
            label_names=config.dataset.labels_str,
            prefix='test_'
        )
        
        # 記錄結束時間和總運行時間
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"結束時間: {end_time}")
        logging.info(f"總運行時間: {duration}")
        logging.info("=== 訓練結束 ===")
        
    except Exception as e:
        logging.error(f"訓練過程中出現錯誤: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()
