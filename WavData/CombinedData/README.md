# 資料夾結構說明

每個子資料夾（例如：P023_吞果凍_2024-12-12_09-33-13）內包含以下檔案：

1. `Probe0_RX_IN_TDM4CH0.wav`
   - 原始音訊檔案
   - 包含錄音的原始數據

2. `[PatientID]_info.json`（例如：P023_info.json）
   - 原始音訊檔案的相關資訊
   - 包含PatientId, selection, score等數據

3. `WavTokenizer_tokens_info.json`
   - 處理後的音訊標記資訊檔案
   - 包含音訊處理的相關參數和設定

4. `WavTokenizer_tokens.npy`
   - 處理後的音訊標記數據檔案
   - 包含轉換後的數值數據

5. `*.png`
   - 音訊視覺化圖表
   - 顯示音訊波形或其他相關視覺化資訊

## 範例資料夾結構
```
P023_吞果凍_2024-12-12_09-33-13/
├── Probe0_RX_IN_TDM4CH0.wav
├── P023_info.json
├── WavTokenizer_tokens_info.json
├── WavTokenizer_tokens.npy
└── [圖表名稱].png
``` 