
# WavFeatureCNN Experiment

## 環境設置 (Environment Setup)

### 1. Google Colab 環境
```bash
# Colab中的路徑
/content/drive/MyDrive/MicforDysphagia/WavData/
```
執行路徑："PYTHONPATH=/content/drive/MyDrive/MicforDysphagia python Model/experiments/baseline/train.py"

### 2. 本地環境
```bash
# 本地環境中的路徑
/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/WavData/
```

### 3. 路徑處理建議
- 使用 `os.path.expanduser()` 處理用戶目錄
- 使用 `os.path.abspath()` 獲取絕對路徑
- 檢查環境變量來判斷運行環境
- 在日誌中記錄完整路徑以便調試

## 數據結構說明 (Data Structure)

1. 原始數據結構 (Raw Data Structure)
/content/drive/MyDrive/MicforDysphagia/WavData/
├── OriginalData/
│   ├── N002/
│   │   └── N002_2024-10-22_10-50-58/
│   │       ├── N002_info.json            # 包含實驗信息
│   │       └── Probe0_RX_IN_TDM4CH0.wav  # 原始音頻文件
│   └── P001/
│       └── P001-1/
│           ├── P001_info.json
│           └── DrySwallow_FB_left_240527.wav
└── ... (其他受試者)

2. 預處理後的數據結構 (Processed Data Structure)
/content/drive/MyDrive/MicforDysphagia/WavData/
└── ProcessedData/
    ├── N002/
    │   └── N002_2024-10-22_10-50-58/
    │       ├── processing_info.json       # 預處理參數信息
    │       ├── original_lps.npy          # 原始特徵
    │       ├── noise_augmented_lps.npy   # 噪聲增強特徵
    │       ├── shift_augmented_lps.npy   # 時移增強特徵
    │       └── mask_augmented_lps.npy    # 頻率遮罩增強特徵
    └── P001/
        └── P001-1/
            ├── processing_info.json
            └── ... (特徵文件，結構同上)


## 運行環境注意事項 (Runtime Environment Notes)

1. Google Colab:
   - 需要先掛載 Google Drive
   - 使用 `/content/drive/MyDrive/` 作為基礎路徑
   - 運行命令：
     ```bash
     PYTHONPATH=$PYTHONPATH:/content/drive/MyDrive/MicforDysphagia python -m Model.experiments.WavFeatureCNN.train
     ```

2. 本地環境:
   - 使用相對路徑或用戶目錄展開
   - 運行命令：
     ```bash
     PYTHONPATH=$PYTHONPATH:. python -m Model.experiments.WavFeatureCNN.train
     ```

3. 路徑驗證:
   - 運行前先確認數據目錄結構
   - 檢查日誌文件中的路徑信息
   - 確保 info 文件和特徵文件都在正確位置

