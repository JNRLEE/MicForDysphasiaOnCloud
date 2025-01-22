各個程式碼檔案的作用:

train_standalone.py:
數據加載與預處理 (load_data):
從指定目錄 (CONFIG['dataset']['data_dir']) 讀取特徵文件 (例如 .npy 檔案) 和對應的 *_info.json 文件。
使用 class_config.py 中的規則 (subject_source, get_action_type, is_class_active) 來確定每個樣本的類別 (e.g., Normal-WaterDrinking, Patient-DrySwallow)。
根據 CLASS_CONFIG 過濾掉未啟用的類別。
對特徵進行標準化處理 (StandardScaler)。
使用 prepare_data 函數將數據分割成訓練集、驗證集和測試集，並確保同一個病人的數據只出現在一個數據集中。
產生並儲存數據分佈資訊文件 (DataDistribution.md)。
模型定義 (WavFeatureCNN): 定義了一個基於全連接層 (FC) 的神經網路模型，用於分類音訊特徵。
模型訓練 (train_model):
使用 tf.data.Dataset 準備訓練和驗證數據。
設定回調函數，包括：
BatchLossCallback: 記錄每個批次的 loss 和 accuracy。
BatchLogger: 記錄每個批次使用的訓練樣本檔案路徑。
AutoMonitor: 自動監控訓練狀況，當驗證集損失在設定的忍耐度內沒有顯著改善時停止訓練。
EarlyStopping: 當驗證集損失在設定的忍耐度內沒有改善時停止訓練，並恢復最佳模型權重。
ReduceLROnPlateau: 當驗證集損失停止改善時降低學習率。
TensorBoard: 將訓練過程中的指標寫入 TensorBoard 日誌。
編譯模型 (model.compile)，使用 Adam 優化器和稀疏類別交叉熵損失函數 (SparseCategoricalCrossentropy)。
開始訓練 (model.fit)。
模型評估 (evaluate_model):
在測試集上評估模型性能。
計算準確率 (accuracy)、混淆矩陣 (confusion matrix)、精確度 (precision)、召回率 (recall) 和 F1 分數 (F1-score)。
將評估結果保存到 JSON 文件 (evaluation_metrics.json) 和 CSV 文件 (confusion_matrix.csv)。
繪製並保存混淆矩陣熱圖 (confusion_matrix.png)。
TSNE 視覺化 (visualize_tsne_results):
讀取 results/tsne_results.csv 檔案 (這個檔案應該是由其他程式生成的 TSNE 降維結果)。
使用 plot_tsne_visualization 函數繪製 2D 和 3D 的 TSNE 視覺化圖，並根據類別和數據集類型 (train/val/test) 對數據點進行著色和標記。


visualize_tsne_custom.py:
TSNE 結果視覺化: 這個檔案的主要目的是根據 class_config.py 的配置，對 results/tsne_results.csv 中的 TSNE 結果進行更精細的視覺化。
數據集信息加載 (load_dataset_info): 從 DataDistribution.md 文件中讀取訓練集、驗證集和測試集的病人 ID 列表。
顏色映射生成 (generate_color_palette): 根據 CLASS_CONFIG 中啟用的類別，為每個類別生成三種不同深淺的顏色，分別對應訓練集、驗證集和測試集。
數據點分類 (classify_data_point): 根據 class_config.py 中的 subject_source, get_action_type, CLASS_CONFIG 以及 DataDistribution.md 中的數據集信息，將每個數據點分類到特定的類別和數據集類型 (例如 Normal-WaterDrinking-train, Patient-DrySwallow-test)。
TSNE 圖繪製 (plot_tsne):
根據數據點的分類和數據集類型，使用 color_map 中的顏色對數據點進行著色。
按特定順序繪製數據點：未啟用/未定義 (最底層) -> 訓練集 -> 驗證集 -> 測試集 (最上層)，以便更清晰地觀察不同數據集和類別的分佈。
生成不同 perplexity 值 (5, 30, 50) 的 2D 和 3D TSNE 圖。


class_config.py:
配置和工具函數: 這個檔案定義了模型訓練和數據處理所需的配置參數和工具函數。
SELECTION_TYPES: 定義了中文實驗類型到英文類別的映射。
SCORE_THRESHOLDS: 定義了用於區分正常人和病人的評分閾值。
CLASS_CONFIG: 定義了哪些類別是啟用的 (1) 或未啟用的 (0)。
SUBJECT_SOURCE_CONFIG: 定義了如何根據病人 ID 前綴 (N 或 P) 和評分來確定病人是正常人還是病人。
subject_source: 根據評分和病人 ID 前綴判斷病人是否為正常人或病人。
get_active_classes: 獲取活動類別列表。
get_active_class_names: 獲取活動類別名稱列表。
get_num_classes: 獲取活動類別數量。
get_class_mapping: 獲取類別到索引的映射。
validate_class_config: 驗證類別配置是否有效。
is_class_active: 檢查類別是否處於激活狀態。
convert_to_one_hot: 將標籤轉換為 one-hot 編碼。
convert_from_one_hot: 將 one-hot 編碼轉換回標籤索引。
update_labels: 更新標籤以匹配活動類別。
filter_data: 根據活動類別過濾數據。
get_classification_mode: 獲取當前的分類模式 (二分類或多分類)。
is_normal: 判斷是否為正常人。
is_patient: 判斷是否為病人。
get_action_type: 將中文實驗類型映射到英文類別。
classify_sample: 根據實驗類型和評分確定完整的分類。
read_info_json: 讀取並解析 info.json 文件。
get_class_from_info: 從 info 字典中獲取類別名稱。