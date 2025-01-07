# Autoencoder Experiment
紀錄東西時請往下紀錄不要覆寫寫料
## 環境設置 (Environment Setup)

### 1. Google Colab 環境
```
# Colab中的路徑
/content/drive/MyDrive/MicforDysphagia/WavData/
```

### 2. 本地環境
```
# 本地環境中的路徑
/Users/[username]/Library/CloudStorage/GoogleDrive-[email]/My Drive/MicforDysphagia/WavData/
```

### 3. 路徑處理建議
- 使用 `os.path.expanduser()` 處理用戶目錄
- 使用 `os.path.abspath()` 獲取絕對路徑
- 檢查環境變量來判斷運行環境
- 在日誌中記錄完整路徑以便調試

## 4. 數據結構說明 (Data Structure)
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
└── AutoEncoderData/
    ├── N002/                                     # 正常受試者目錄
    │   └── N002_乾吞嚥1口_2024-10-22_10-50-58/    # 實驗目錄
    │       └── Probe0_RX_IN_TDM4CH0_tokens.npy   # token數據
    │       └── N002_tokens_info.json             # info文件
    └── P001/                                     # 病患受試者目錄
        └── P001_乾吞嚥1口_2024-12-02_09-43-15/    # 實驗目錄
            └── Probe0_RX_IN_TDM4CH0_tokens.npy   # token數據文件
    │       └── N002_tokens_info.json             # info文件

## 5. 數據要求 (Data Requirements)

### 1. Info文件 (Info Files)
- 位置：在原始數據目錄中（不在AutoEncoderData中）
- 命名格式：`*_info.json` 或 `*_info`
- 內容格式：JSON格式（可以有換行或無換行）
- 必需字段：`score`（用於篩選受試者）, 'selection'


### 2. Token文件 (Token Files)
- 位置：在AutoEncoderData目錄下的對應受試者實驗目錄中
- 文件名：`Probe0_RX_IN_TDM4CH0_tokens.npy`
- 內容：包含 `discrete_code` 字段的numpy數組

### 3. 受試者篩選標準 (Subject Selection Criteria)
- 正常受試者 (Normal): score = 0
- 病人受試者 (Patient): score > 3

### 4. 實驗類型 (Experiment Type)
- 僅處理 DrySwallow 實驗數據（目錄名包含"乾吞嚥"）
- 其他類型的實驗數據會被忽略

## 數據加載過程 (Data Loading Process)

1. 從AutoEncoderData目錄加載token數據
2. 從原始數據目錄讀取info文件以獲取score
3. 根據score篩選符合條件的受試者
4. 數據集分割比例：
   - 訓練集：70%
   - 驗證集：15%
   - 測試集：15%

## 注意事項 (Important Notes)

1. Info文件讀取：
   - 支持兩種文件格式：`*_info.json` 和 `*_info`
   - 支持有換行和無換行的JSON格式
   - 文件編碼必須為UTF-8

2. 目錄結構：
   - AutoEncoderData和原始數據必須保持相同的目錄層級結構
   - 受試者ID必須在兩個位置保持一致

3. 錯誤處理：
   - 所有錯誤和警告都會記錄在 `logs/autoencoder_dataloader.log`
   - 控制台只顯示警告和錯誤信息

## 配置文件 (Configuration)

在 `config.yaml` 中：
```yaml
data_dir: "WavData/AutoEncoderData"  # 必須指向AutoEncoderData目錄
batch_size: 32                       # 批次大小
```

## 運行環境注意事項 (Runtime Environment Notes)

1. Google Colab:
   - 需要先掛載 Google Drive
   - 使用 `/content/drive/MyDrive/` 作為基礎路徑
   - 運行命令：
     ```bash
     PYTHONPATH=$PYTHONPATH:/content/drive/MyDrive/MicforDysphagia python -m Model.experiments.autoencoder.train
     ```

2. 本地環境:
   - 使用相對路徑或展開的用戶目錄路徑
   - 運行命令：
     ```bash
     PYTHONPATH=$PYTHONPATH:. python -m Model.experiments.autoencoder.train
     ```

3. 路徑驗證:
   - 運行前先確認數據目錄結構
   - 檢查日誌文件中的路徑信息
   - 確保 info 文件和 token 文件都在正確位置

---

## 更新記錄 (Update Log)



### 2024-01-04 更新 (2)
colab的路徑
#### 已解決的問題
1. 數據維度處理
   - 修正了3維數據的處理邏輯，通過取第一個樣本將其轉換為2維數據
   - 修正了padding的維度匹配問題，確保與token_data維度一致

2. 訓練過程
   - 成功完成了22個epoch的訓練
   - 觀察到訓練損失和驗證損失基本穩定
   - 學習率在第18個epoch時從0.001降低到0.0005

#### 當前問題
1. 模型效果
   - 損失值較大（約6.93M），可能需要：
     - 檢查數據的預處理和標準化
     - 調整模型架構
     - 優化超參數設置

2. 數據處理策略
   - 需要重新考慮固定長度768的限制
   - 需要深入分析token.npy的數據結構和維度含義
   - 需要評估當前的數據提取方式是否合適

#### 下一步計劃
1. 分析token.npy的數據結構
2. 實現動態長度的數據處理
3. 優化數據預處理和標準化流程 

### 2024-01-04 更新 (3)
#### 已完成的改進
1. 改進了數據加載邏輯
   - 現在使用 `features` 而不是 `discrete_code` 進行訓練
   - 添加了序列長度標準化處理（填充/截斷到固定長度2048）
   - 改進了數據形狀檢查和驗證

2. 優化了模型架構
   - 簡化為純卷積層設計
   - 輸入/輸出維度：(512, 2048)
   - 使用對稱的編碼器-解碼器結構
   - 添加了適當的填充以保持特徵圖大小

3. 訓練改進
   - 實現了更穩定的訓練過程
   - 添加了學習率自適應調整
   - 改進了損失函數的監控

#### 注意事項
- 數據預處理時會將序列長度標準化為2048
- 較長的序列會從中間截取
- 較短的序列會使用零填充
- 模型現在能更好地處理可變長度的輸入數據

# Info 文件規格說明（⚠️ 極其重要！請仔細閱讀！）

## 1. Info 文件位置
- ⚠️ Info 文件必須位於原始數據目錄（不是 AutoEncoderData）中
- ⚠️ Info 文件必須與實驗數據在同一個目錄下

## 2. Info 文件命名規則
- 格式1：`實驗ID_info.json`
- 格式2：`實驗ID_info`
- 示例：如果實驗ID是 N001，則文件名應為 `N001_info.json` 或 `N001_info`

## 3. Info 文件內容格式
- 必須是 JSON 格式
- 可以有換行或無換行
- 編碼必須是 UTF-8

## 4. Info 文件必需字段
1. `score`：用於篩選受試者
   - 正常人：必須是 0
   - 病人：必須大於 3

2. `selection`：實驗類型
   必須是以下其中之一：
   ```
   無動作, 無吞嚥              -> NoMovement（無動作類型）
   乾吞嚥1口, 乾吞嚥2口,       -> DrySwallow（乾吞嚥類型）
   乾吞嚥3口, 乾吞嚥
   餅乾1塊, 餅乾2塊, 餅乾      -> Cracker（餅乾類型）
   吞果凍, 果凍                -> Jelly（果凍類型）
   吞水10ml, 吞水20ml,         -> WaterDrinking（喝水類型）
   喝水, 吞水
   ```

## 5. Info 文件示例
```json
{
    "score": 0,
    "selection": "乾吞嚥"
}
```

## 6. 重要提醒（⚠️）
- Info 文件缺失或格式錯誤會導致數據被跳過
- score 值不符合標準會導致數據被跳過
- selection 值不在上述列表中會導致數據被跳過

### 2024-01-05 更新
#### 常見問題和混淆原因總結（⚠️ 重要！）
1. Info 文件相關問題
   - 混淆原因：誤以為 info 文件應該在 AutoEncoderData 目錄中
   - 正確位置：必須在原始數據目錄（WavData/NormalSubject 或 WavData/PatientSubject）中
   - 解決方法：始終檢查原始數據目錄，而不是 AutoEncoderData 目錄

2. 數據加載問題
   - 混淆原因：誤將 `discrete_code` 與 `features` 混淆
   - 正確用法：應該使用 `features` 字段進行訓練
   - 解決方法：確保從 .npy 文件中讀取 `features` 字段

3. 實驗類型篩選問題
   - 混淆原因：過度限制實驗類型（只看"乾吞嚥"）
   - 正確做法：應該接受所有 SELECTION_TYPES 中定義的 DrySwallow 類型
   - 解決方法：使用完整的實驗類型列表進行匹配

4. 樣本數量不足問題
   - 混淆原因：忽略了部分有效數據
   - 解決方法：移除不必要的過濾條件，確保所有有效數據都被加載

#### 今日訓練結果
1. 數據統計
   - 總樣本數：90個
   - 正常人樣本：29個
   - 病人樣本：61個
   - 說話者分布：9位正常人，22位病人

2. 數據分割
   - 訓練集：21位說話者（62個樣本，正常人19個，病人43個）
   - 驗證集：3位說話者（9個樣本，正常人4個，病人5個）
   - 測試集：7位說話者（19個樣本，正常人6個，病人13個）

3. 訓練過程
   - 訓練輪數：6輪（Early Stopping）
   - 最佳驗證損失：0.3712（第1輪）
   - 最終測試結果：
     * 準確率：68.42%
     * 精確率：68.42%
     * 召回率：100%
     * F1分數：81.25%

4. 模型保存位置
   - 檢查點目錄：`Model/experiments/autoencoder/checkpoints/`
   - 最佳模型：`Model/experiments/autoencoder/checkpoints/model_01.weights.h5`
   - 訓練日誌：`Model/experiments/autoencoder/logs/`

5. 問題分析
   - 模型在訓練集上表現穩定，但驗證損失波動較大
   - 存在過擬合趨勢，驗證損失在第一輪後開始上升
   - 召回率很高但精確率較低，說明模型傾向於將樣本預測為病人
   - 測試集中完全沒有正確識別出正常人（TN=0）

6. 下一步建議
   - 增加數據增強以改善泛化能力
   - 調整模型架構，可能需要更深的網絡
   - 考慮添加正則化以減少過擬合
   - 平衡數據集，可能需要對正常人樣本進行上採樣

## 警告
⚠️ 本項目包含關鍵的數據讀取路徑和邏輯，以下路徑和文件結構不要隨意修改，如需更改請先諮詢負責人：
- 原始音頻文件（.wav）
- 特徵文件（.npy）
- 信息文件（.json）
- 數據目錄結構