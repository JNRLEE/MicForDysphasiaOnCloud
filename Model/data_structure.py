"""
這個檔案平時不要亂修改，如果要修改請先詢問
數據集結構說明：

1. 原始數據結構 (Raw Data Structure)
/content/drive/MyDrive/MicforDysphagia/WavData/
├── NormalSubject/
│   ├── N002/
│   │   └── N002_2024-10-22_10-50-58/
│   │       ├── N002_info.json(或是N002_info)           # 包含實驗信息
│   │       └── Probe0_RX_IN_TDM4CH0.wav # 原始音頻文件
│   └── ... (其他正常受試者)
└── PatientSubject/
  ├── P001/
  │   └── P001-1/
  │       ├── P001_info.json(或是P001_info)
  │       └── DrySwallow_FB_left_240527.wav
  └── ... (其他病患受試者)
（info說明：
# Update selection types to include all possible Chinese descriptions
selection_types = {
    'NoMovement': ["無動作", "無吞嚥"],
    'DrySwallow': ["乾吞嚥1口", "乾吞嚥2口", "乾吞嚥3口", "乾吞嚥"],
    'Cracker': ["餅乾1塊", "餅乾2塊", "餅乾"],
    'Jelly': ["吞果凍", "果凍"],
    'WaterDrinking': ["吞水10ml", "吞水20ml", "喝水", "吞水"]
}）

2. 預處理後的數據結構 (Processed Data Structure)
/content/drive/MyDrive/MicforDysphagia/WavData/
└── ProcessedData/
  ├── N002/
  │   └── N002_2024-10-22_10-50-58/
  │       └── Probe0_RX_IN_TDM4CH0/
  │           ├── processing_info.json      # 預處理參數信息
  │           ├── original_lps.npy        # 原始特徵
  │           ├── noise_augmented_lps.npy # 噪聲增強特徵
  │           ├── shift_augmented_lps.npy # 時移增強特徵
  │           └── mask_augmented_lps.npy  # 頻率遮罩增強特徵
  └── P001/
    └── P001-1/
      └── DrySwallow_FB_left_240527/
        ├── processing_info.json
        └── ... (特徵文件，結構同上)

3. 文件對應關係：
- 每個原始 .wav 文件都對應一個處理後的目錄
- 每個處理後的目錄包含該音頻的四種特徵表示
- processing_info.json 記錄預處理參數：
  {
  'original_shape': [樣本數, 特徵維度, 時間步長],
  'sr': 16000,
  'frame_size': 256,
  'overlap': 128,
  'fft_size': 256
  }

4. 標籤對應：
- 正常受試者(Normal): 0-4
  - NoMovement: 0
  - DrySwallow: 1
  - Cracker: 2
  - Jelly: 3
  - WaterDrinking: 4
- 病患受試者(Patient): 5-9
  - NoMovement: 5
  - DrySwallow: 6
  - Cracker: 7
  - Jelly: 8
  - WaterDrinking: 9
"""
