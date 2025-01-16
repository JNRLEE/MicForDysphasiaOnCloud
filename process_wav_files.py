# 此程式用於處理WAV檔案並生成WavTokenizer特徵
# 將自動處理ProcessingSpace目錄下的所有WAV檔案

import os
import sys
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime

# Check for required packages
def check_requirements():
    required_packages = ['einops', 'torch', 'torchaudio', 'numpy', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

# Check requirements before importing other modules
check_requirements()

from tqdm import tqdm  # 添加進度條

# 添加WavTokenizer目錄到Python路徑
current_dir = Path(__file__).parent.absolute()
wavtokenizer_dir = current_dir / "WavTokenizer"
sys.path.append(str(wavtokenizer_dir))

from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer

def process_wav_file(wav_path: Path, wavtokenizer: WavTokenizer, device: torch.device) -> tuple:
    """
    處理單個WAV檔案並生成特徵和離散碼
    
    Args:
        wav_path: WAV檔案路徑
        wavtokenizer: WavTokenizer模型
        device: 運算設備
        
    Returns:
        tuple: (features, discrete_code, sample_rate)
    """
    try:
        # 載入WAV檔案
        wav, sr = torchaudio.load(str(wav_path))
        wav = convert_audio(wav, sr, 24000, 1)
        bandwidth_id = torch.tensor([0])
        wav = wav.to(device)
        
        # 生成特徵和離散碼
        features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        return features, discrete_code, sr
    except Exception as e:
        print(f"\n處理WAV檔案時發生錯誤 {wav_path}: {str(e)}")
        raise

def save_tokens_info(save_path: Path, original_file: str, feature_shape: tuple, token_shape: tuple, sample_rate: int):
    """
    儲存tokens_info.json檔案
    
    Args:
        save_path: 儲存路徑
        original_file: 原始WAV檔案路徑
        feature_shape: 特徵形狀
        token_shape: token形狀
        sample_rate: 採樣率
    """
    info = {
        "original_file": f"./{original_file.name}",  # 使用相對路徑
        "sample_rate": sample_rate,
        "token_shape": list(token_shape),  # 轉換為list以便JSON序列化
        "feature_shape": list(feature_shape),  # 轉換為list以便JSON序列化
        "processing_time": datetime.now().isoformat()
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def main():
    try:
        # 設定裝置
        device = torch.device('cpu')
        print("使用CPU進行處理")
        
        # 載入WavTokenizer模型
        config_path = str(wavtokenizer_dir / "wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
        model_path = str(wavtokenizer_dir / "wavtokenizer_medium_music_audio_320_24k_v2.ckpt")
        
        print(f"載入配置文件: {config_path}")
        print(f"載入模型文件: {model_path}")
        
        wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        wavtokenizer = wavtokenizer.to(device)
        print("模型載入完成")
        
        # 設定ProcessingSpace目錄路徑
        processing_space = Path("WavData/Combined/ProcessingSpace")
        
        # 獲取所有需要處理的目錄
        subject_dirs = [d for d in processing_space.iterdir() if d.is_dir() and not d.name.startswith('.')]
        print(f"\n找到 {len(subject_dirs)} 個待處理目錄")
        
        # 使用tqdm顯示總體進度
        for subject_dir in tqdm(subject_dirs, desc="處理目錄進度"):
            print(f"\n開始處理目錄: {subject_dir.name}")
            
            # 尋找WAV檔案
            wav_files = list(subject_dir.glob("*.wav"))
            if not wav_files:
                print(f"在 {subject_dir.name} 中找不到WAV檔案")
                continue
            
            print(f"找到 {len(wav_files)} 個WAV檔案")
            
            # 處理每個WAV檔案
            for wav_file in tqdm(wav_files, desc="處理檔案進度", leave=False):
                try:
                    print(f"\n處理檔案: {wav_file.name}")
                    
                    # 檢查是否已經處理過
                    tokens_path = subject_dir / "WavTokenizer_tokens.npy"
                    info_path = subject_dir / "WavTokenizer_tokens_info.json"
                    if tokens_path.exists() and info_path.exists():
                        print(f"檔案已處理過，跳過: {wav_file.name}")
                        continue
                    
                    # 生成特徵和離散碼
                    features, discrete_code, sample_rate = process_wav_file(wav_file, wavtokenizer, device)
                    
                    # 轉換為numpy數組並獲取形狀
                    features_np = features.cpu().numpy()
                    discrete_code_np = discrete_code.cpu().numpy()
                    
                    # 儲存tokens.npy
                    np.save(str(tokens_path), {
                        'features': features_np,
                        'tokens': discrete_code_np
                    })
                    
                    # 儲存tokens_info.json
                    save_tokens_info(
                        info_path,
                        wav_file,
                        features_np.shape,
                        discrete_code_np.shape,
                        sample_rate
                    )
                    
                    print(f"已成功處理並儲存: {wav_file.name}")
                    print(f"特徵形狀: {features_np.shape}")
                    print(f"離散碼形狀: {discrete_code_np.shape}")
                    
                except Exception as e:
                    print(f"處理 {wav_file.name} 時發生錯誤: {str(e)}")
                    continue
        
        print("\n所有處理完成!")
        
    except Exception as e:
        print(f"\n程式執行時發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()