import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
import gc
import sys
from pathlib import Path

class SwallowingDataEvaluator:
    def __init__(self):
        """初始化評估器"""
        # 使用固定的Google Drive掛載點
        self.drive_path = '/content/drive/MyDrive/MicforDysphagia'
        if not os.path.exists(self.drive_path):
            raise RuntimeError(f"Google Drive path not found: {self.drive_path}")

        # 更新所有路徑以使用drive_path
        self.model_path = os.path.join(self.drive_path, 'TrainedModel', 'my_trained_model_1129_simple.keras')
        self.processed_data_path = os.path.join(self.drive_path, 'ProcessedData')
        self.results_path = os.path.join(self.drive_path, 'ResultFig', 'tsne_results_with1130keras.csv')
        
        # 驗證路徑
        self._verify_paths()
        
        self.model = None
        self.results_df = None

    def _verify_paths(self):
        """驗證所有必要的路徑"""
        paths_to_check = {
            'Model file': self.model_path,
            'Processed data directory': self.processed_data_path,
            'Results file': Path(self.results_path).parent
        }
        
        for name, path in paths_to_check.items():
            if not os.path.exists(path):
                print(f"Warning: {name} not found at {path}")
                if name == 'Results file':
                    # 如果是結果文件的目錄不存在，創建它
                    os.makedirs(path, exist_ok=True)
                    print(f"Created directory for results: {path}")
                else:
                    raise FileNotFoundError(f"{name} not found at {path}")

    def load_model_and_data(self):
        """載入模型和CSV結果檔"""
        print("Loading model and data...")
        try:
            print(f"Attempting to load model from: {self.model_path}")
            self.model = load_model(self.model_path)
            print("Model loaded successfully")
            
            if os.path.exists(self.results_path):
                self.results_df = pd.read_csv(self.results_path)
                print("Results CSV loaded successfully")
            else:
                print("Creating new results DataFrame")
                self.results_df = pd.DataFrame(columns=['file_path', 'action_type', 'selection', 'score', 'label', 'tsne_x', 'tsne_y'])
            
        except Exception as e:
            print(f"Error during loading: {str(e)}")
            raise

    def get_processed_file_path(self, original_path):
        """將原始WAV文件路徑轉換為處理後的NPY文件路徑"""
        try:
            # 從原始路徑提取相關部分
            parts = original_path.split('WavData/')[-1].split('/')
            # 直接取ID部分（N002或P001等）
            subject_id = parts[1]  # 獲取ID部分
            relative_path = '/'.join(parts[2:])  # 獲取剩餘路徑
            wav_filename = os.path.basename(original_path)
            wav_name = os.path.splitext(wav_filename)[0]
            
            # 構建新路徑：/ProcessedData/N002/... 或 /ProcessedData/P001/...
            npy_path = os.path.join(self.processed_data_path, 
                                  subject_id,
                                  os.path.dirname(relative_path),
                                  wav_name,
                                  'original_lps.npy')
            
            print(f"Looking for file at: {npy_path}")  # debug輸出
            
            if not os.path.exists(npy_path):
                print(f"Warning: Processed file not found: {npy_path}")
                return None
                
            return npy_path
        except Exception as e:
            print(f"Error in path conversion: {str(e)}")
            return None

    def calculate_loss(self, features, true_label):
        """計算categorical crossentropy loss"""
        try:
            # 確保特徵格式正確
            if len(features.shape) == 3:
                features = features[:, :, :, np.newaxis]
            
            # 獲取預測
            predictions = self.model.predict(features, verbose=0)
            
            # 檢查預測值是否確實經過 softmax (應該要加總為1)
            print("\nDEBUG - Predictions check:")
            print(f"Random sample prediction sums: {np.sum(predictions[0]):.6f}")  # 應該接近 1.0
            print(f"Min prediction value: {np.min(predictions):.6f}")  # 應該 >= 0
            print(f"Max prediction value: {np.max(predictions):.6f}")  # 應該 <= 1
            
            # 準備真實標籤的 one-hot 編碼
            true_label_one_hot = np.zeros(10)
            true_label_one_hot[true_label] = 1
            true_label_one_hot = true_label_one_hot[np.newaxis, :]
            true_label_one_hot = np.repeat(true_label_one_hot, predictions.shape[0], axis=0)
            
            # 檢查 one-hot 編碼
            print("\nDEBUG - One-hot encoding check:")
            print(f"True label: {true_label}")
            print(f"One-hot shape: {true_label_one_hot.shape}")
            print(f"One-hot sample: {true_label_one_hot[0]}")
            
            # 計算 loss
            loss = categorical_crossentropy(
                true_label_one_hot, 
                predictions, 
                from_logits=False
            ).numpy()
            
            # 檢查 loss 值
            print("\nDEBUG - Loss values check:")
            print(f"Loss shape: {loss.shape}")
            print(f"Sample loss values: {loss[:5]}")
            
            mean_loss = float(np.mean(loss))
            print(f"Mean loss: {mean_loss:.6f}")
            
            return mean_loss
            
        except Exception as e:
            print(f"\nError in calculate_loss:")
            print(f"Error message: {str(e)}")
            print(f"Features shape: {features.shape}")
            if 'predictions' in locals():
                print(f"Predictions shape: {predictions.shape}")
            return None

    def process_all_files(self):
        """處理所有文件並計算loss"""
        print("Starting to process all files...")
        
        if 'loss' not in self.results_df.columns:
            self.results_df['loss'] = np.nan
        
        for idx, row in self.results_df.iterrows():
            try:
                processed_path = self.get_processed_file_path(row['file_path'])
                if processed_path is None:
                    continue

                # 載入特徵並檢查維度
                features = np.load(processed_path)
                print(f"Loaded features shape: {features.shape}")
                
                # 確保特徵格式正確
                if len(features.shape) == 3:
                    features = features[:, :, :, np.newaxis]
                print(f"Processed features shape: {features.shape}")

                loss = self.calculate_loss(features, row['label'])
                if loss is not None:
                    self.results_df.loc[idx, 'loss'] = loss
                    print(f"Calculated loss: {loss}")

                if idx % 10 == 0:
                    print(f"Processed {idx} files...")
                    self.results_df.to_csv(self.results_path, index=False)
                    gc.collect()

            except Exception as e:
                print(f"Error processing file {row['file_path']}: {str(e)}")
                continue

        # 最終保存結果
        self.results_df.to_csv(self.results_path, index=False)
        print("Processing completed!")

def main():
    try:
        evaluator = SwallowingDataEvaluator()
        evaluator.load_model_and_data()
        evaluator.process_all_files()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
