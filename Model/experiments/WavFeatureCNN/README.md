
# WavFeatureCNN Experiment

## 環境設置 (Environment Setup)

### 1. Google Colab 環境
```bash
# Colab中的路徑
/content/drive/MyDrive/MicforDysphagia/WavData/
```

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

3. 訓練數據目錄 (Training Data Directory)
/content/drive/MyDrive/MicforDysphagia/WavData/
└── WavFeatureCNNData/
    ├── NormalSubject/                            # 正常受試者目錄
    ├── PatientSubject/                           # 病人受試者目錄
    └── ... (其他受試者)

### 4. 受試者篩選標準 (Subject Selection Criteria)
- 正常受試者 (Normal): score = 0
- 病人受試者 (Patient): score > 3

### 5. 實驗類型 (Experiment Type)
- 僅處理 DrySwallow 實驗數據（目錄名包含"乾吞嚥"）
- 其他類型的實驗數據會被忽略

## 數據加載過程 (Data Loading Process)

1. 從WavFeatureCNNData目錄加載特徵數據
2. 從原始數據目錄讀取info文件以獲取score和patient_id
3. 根據score篩選符合條件的受試者
4. 創建滑動窗口分割數據，形狀為 `[512, window_size]` 並有一定的重疊
5. 數據集分割比例：
   - 訓練集：70%
   - 驗證集：15%
   - 測試集：15%

## 模型結構 (Model Architecture)

使用簡化的 CNN 結構，專注於捕捉局部時間特徵，並使用全局平均池化來降低參數量和防止過擬合。

## 訓練過程 (Training Process)

1. 加載並預處理數據（滑動窗口分割）
2. 創建和編譯模型
3. 設置回調函數（模型檢查點和 TensorBoard）
4. 執行訓練過程
5. 評估模型並生成混淆矩陣
6. 保存訓練歷史和最佳模型

## 注意事項 (Important Notes)

1. Info文件讀取：
   - 支持兩種文件格式：`*_info.json` 和 `*_info`
   - 支持有換行和無換行的JSON格式
   - 文件編碼必須為UTF-8

2. 目錄結構：
   - WavFeatureCNNData和原始數據必須保持相同的目錄層級結構
   - 受試者ID必須在兩個位置保持一致

3. 錯誤處理：
   - 所有錯誤和警告都會記錄在 `logs/WavFeatureCNN_dataloader.log`
   - 控制台只顯示警告和錯誤信息

## 配置文件 (Configuration)

在 `config.yaml` 中：

```yaml
data_dir: "WavData/WavFeatureCNNData"  # 必須指向WavFeatureCNNData目錄
batch_size: 32                       # 批次大小
feature_dim: 512                     # 窗口大小
window_overlap: 256                  # 窗口重疊大小
window_size: 512                     # 窗口大小
```

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

---

## 更新記錄 (Update Log)

### 2025-01-13 更新

#### 已完成的改進
1. 禁用 k-means 功能，改用滑動窗口分割來處理特徵數據。
2. 調整數據加載器，實現帶有重疊的滑動窗口分割。
3. 簡化 CNN 模型結構，減少卷積層數量並使用較大的卷積核。
4. 更新配置文件以反映新的數據處理和模型參數。

#### 注意事項
- 現在模型僅接受滑動窗口分割後的特徵數據 `[512, window_size]` 作為輸入。
- 移除了 K-Means 壓縮特徵的步驟，直接使用滑動窗口分割後的數據進行訓練。
- 確保所有數據窗口的特徵形狀為 `[512, window_size]`，否則會被跳過。
- 調整了數據加載和分割邏輯，以適應滑動窗口分割後的數據結構。

