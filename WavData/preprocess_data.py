import os
import numpy as np
import json
from tqdm import tqdm
import gc

# Add your custom module path
import sys  # 加入這行以匯入 sys 模組

sys.path.append('/content/drive/MyDrive/MicforDysphagia/')
from Function import utils

class DataPreprocessor:
    def __init__(self, sr=16000, frame_size=256, overlap=128, fft_size=256):
        self.sr = sr
        self.frame_size = frame_size
        self.overlap = overlap
        self.fft_size = fft_size
        self.seq_num = overlap + 1
        print(f"Initializing DataPreprocessor with parameters:")
        print(f"- Sampling rate: {sr}")
        print(f"- Frame size: {frame_size}")
        print(f"- Overlap: {overlap}")
        print(f"- FFT size: {fft_size}")

    def process_single_file(self, wav_path, save_dir):
        """處理單個音頻文件並保存其特徵"""
        print(f"\nProcessing file: {wav_path}")
        try:
            # 建立保存路徑
            file_name = os.path.splitext(os.path.basename(wav_path))[0]
            save_path = os.path.join(save_dir, file_name)
            os.makedirs(save_path, exist_ok=True)
            print(f"Save path created: {save_path}")

            # 載入和處理音頻
            print("Loading WAV file...")
            X_wav = utils.load_wavs([wav_path], self.sr)
            print("Extracting LPS features...")
            X_lps, _ = utils.lps_extract(X_wav, self.frame_size, 
                                       self.overlap, self.fft_size, to_list=False)
            X_lps = X_lps.T
            X_lps = utils.time_splite(X_lps, time_len=self.seq_num, padding=True)
            print(f"Feature shape after processing: {X_lps.shape}")

            # 生成增強數據
            print("Generating augmented data...")
            print("- Adding noise...")
            X_noise = self.add_noise(X_lps.copy())
            print("- Applying time shift...")
            X_shift = self.time_shift(X_lps.copy())
            print("- Applying frequency mask...")
            X_mask = self.frequency_mask(X_lps.copy())

            # 分別保存每種特徵
            print("Saving features...")
            np.save(os.path.join(save_path, 'original_lps.npy'), X_lps)
            np.save(os.path.join(save_path, 'noise_augmented_lps.npy'), X_noise)
            np.save(os.path.join(save_path, 'shift_augmented_lps.npy'), X_shift)
            np.save(os.path.join(save_path, 'mask_augmented_lps.npy'), X_mask)
            print("Features saved successfully")

            # 保存處理信息
            info = {
                'original_shape': X_lps.shape,
                'sr': self.sr,
                'frame_size': self.frame_size,
                'overlap': self.overlap,
                'fft_size': self.fft_size
            }
            with open(os.path.join(save_path, 'processing_info.json'), 'w') as f:
                json.dump(info, f)
            print("Processing info saved")

            # 清理記憶體
            del X_wav, X_lps, X_noise, X_shift, X_mask
            gc.collect()
            print("Memory cleaned")

            return True

        except Exception as e:
            print(f"ERROR processing {wav_path}: {str(e)}")
            return False

    @staticmethod
    def add_noise(data, noise_factor=0.005):
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return data + noise_factor * noise

    @staticmethod
    def time_shift(data, shift_range=2):
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(data, shift, axis=1)

    @staticmethod
    def frequency_mask(data, max_mask_size=8):
        mask_size = np.random.randint(1, max_mask_size)
        mask_start = np.random.randint(0, data.shape[1] - mask_size)
        data_masked = data.copy()
        data_masked[:, mask_start:mask_start+mask_size] = 0
        return data_masked

    def process_directory(self, input_dir, output_base_dir):
        """處理整個目錄的音頻文件"""
        print(f"\nProcessing directory: {input_dir}")
        print(f"Output directory: {output_base_dir}")
        
        processed_count = 0
        error_count = 0
        
        # 遍歷目錄
        for root, _, files in os.walk(input_dir):
            wav_files = [f for f in files if f.endswith('.wav') and not f.startswith('._')]
            
            if not wav_files:
                print(f"No WAV files found in {root}")
                continue

            print(f"\nFound {len(wav_files)} WAV files in {root}")
            
            # 創建對應的輸出目錄結構
            rel_path = os.path.relpath(root, input_dir)
            output_dir = os.path.join(output_base_dir, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

            # 處理每個文件
            for wav_file in tqdm(wav_files, desc=f"Processing files in {rel_path}"):
                wav_path = os.path.join(root, wav_file)
                success = self.process_single_file(wav_path, output_dir)
                
                if success:
                    processed_count += 1
                    print(f"Successfully processed: {wav_file}")
                else:
                    error_count += 1
                    print(f"Failed to process: {wav_file}")

                # 定期清理記憶體
                if processed_count % 10 == 0:
                    print(f"Progress: {processed_count} files processed, {error_count} errors")
                    gc.collect()

        return processed_count, error_count

def main():
    print("=== Starting Audio Preprocessing ===")
    
    # Google Drive 路徑設置
    input_dirs = [
        '/content/drive/MyDrive/MicforDysphagia/WavData/NormalSubject',
        '/content/drive/MyDrive/MicforDysphagia/WavData/PatientSubject'
    ]
    output_base_dir = '/content/drive/MyDrive/MicforDysphagia/WavData/ProcessedData'
    
    print("\nInput directories:")
    for dir in input_dirs:
        print(f"- {dir}")
    print(f"Output directory: {output_base_dir}")

    # 創建處理器實例
    print("\nInitializing processor...")
    processor = DataPreprocessor()

    # 處理每個輸入目錄
    total_processed = 0
    total_errors = 0

    for input_dir in input_dirs:
        print(f"\n{'='*50}")
        print(f"Processing directory: {input_dir}")
        processed, errors = processor.process_directory(input_dir, output_base_dir)
        total_processed += processed
        total_errors += errors
        print(f"Directory complete - Processed: {processed}, Errors: {errors}")

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total files processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print("=== Preprocessing Finished ===")

if __name__ == "__main__":
    main()
