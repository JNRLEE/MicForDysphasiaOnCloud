# 三元組損失模型

這個模型使用三元組損失（Triplet Loss）來訓練一個深度學習模型，用於吞嚥聲音的分類。

## 模型架構

模型由以下幾個主要部分組成：

1. CNN基礎網絡：用於提取音頻特徵
2. 嵌入層：將特徵映射到低維空間
3. 三元組損失層：計算錨點、正例和負例之間的距離

## 文件結構

- `config.py`: 定義模型配置類
- `config_colab.yaml`: YAML格式的配置文件
- `model.py`: 實現三元組損失模型
- `main.py`: 主程序，用於訓練和評估模型

## 使用方法

1. 修改配置文件 `config_colab.yaml`，設置合適的參數：
   ```yaml
   input_height: 128
   input_width: 128
   input_channels: 1
   filters: [32, 64, 128, 256, 512]
   kernel_sizes: [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
   dilation_rates: [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
   dropout_rate: 0.3
   embedding_size: 128
   learning_rate: 0.001
   margin: 0.5
   save_dir: "/content/drive/MyDrive/MicforDysphagia/saved_models"
   ```

2. 運行訓練程序：
   ```bash
   python main.py
   ```

3. 訓練完成後，模型和訓練歷史將保存在指定目錄。

## 數據格式

- 輸入數據應為LPS（Log Power Spectrogram）特徵
- 每個音頻文件會被分割成多個片段
- 模型會自動聚合同一音頻文件的所有片段進行預測

## 注意事項

1. 確保數據目錄結構正確
2. 根據實際需求調整配置參數
3. 監控訓練過程中的損失值變化 