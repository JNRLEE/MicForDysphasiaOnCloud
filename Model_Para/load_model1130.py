

# Define function for file-level prediction
def predict_wav_file(model, wav_path):
    """Predict a single WAV file"""
    X_wav = utils.load_wavs([wav_path], sr)
    X_lps, _ = utils.lps_extract(X_wav, frame_size, overlap, fft_size, to_list=False)
    X_lps = X_lps.T
    X_lps = utils.time_splite(X_lps, time_len=seq_num, padding=False)
    X_lps = X_lps[:, :, :, np.newaxis]

    predictions = model.predict(X_lps)
    final_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(final_prediction)

    return predicted_class, final_prediction

def evaluate_files(model, file_paths, true_labels):
    """Evaluate model performance on file level"""
    predictions = []
    for wav_path in file_paths:
        pred_class, _ = predict_wav_file(model, wav_path)
        predictions.append(pred_class)

    accuracy = np.mean(np.array(predictions) == true_labels)
    return accuracy, predictions

# Separate outside test data first
outside_test_ratio = 0.2
all_wav_files = []
all_labels = []

for label, folder_info in data_folders.items():
    base_path = folder_info['path']
    selection_type = folder_info['type']
    wav_files = get_wav_files_recursive(base_path, selection_type)

    for wav_file in wav_files:
        all_wav_files.append(wav_file)
        all_labels.append(label)

# Split files for outside testing
train_files, test_files, train_labels, test_labels = train_test_split(
    all_wav_files, all_labels,
    test_size=outside_test_ratio,
    random_state=42,
    stratify=all_labels
)

# Process training files
X_lps_list = []
X_info_list = []
processed_classes = []

for label in np.unique(train_labels):
    label_files = [f for f, l in zip(train_files, train_labels) if l == label]
    print(f"\nProcessing class {label}")
    print(f"Found {len(label_files)} files")

    if not label_files:
        continue

    try:
        X_wavs = utils.load_wavs(label_files, sr)
        X_lps, X_info = utils.lps_extract(X_wavs, frame_size, overlap, fft_size, to_list=False)
        X_lps = X_lps.T
        X_lps = utils.time_splite(X_lps, time_len=seq_num, padding=False)
        X_lps = X_lps[:, :, :, np.newaxis]

        X_lps_list.append(X_lps)
        X_info_list.extend(X_info)
        processed_classes.append(label)

    except Exception as e:
        print(f"Error processing class {label}: {e}")

# Modified data loading section with better error handling
X_lps_list = []
X_info_list = []
processed_classes = []

for label, folder_info in data_folders.items():
    base_path = folder_info['path']
    selection_type = folder_info['type']

    wav_files = get_wav_files_recursive(base_path, selection_type)
    print(f"\nProcessing class {label} ({selection_type})")
    print(f"Found {len(wav_files)} files")

    if not wav_files:
        print(f"Warning: No files found for class {label} ({selection_type})")
        continue

    try:
        X_wavs = utils.load_wavs(wav_files, sr)
        X_lps, X_info = utils.lps_extract(X_wavs, frame_size, overlap, fft_size, to_list=False)
        X_lps = X_lps.T
        X_lps = utils.time_splite(X_lps, time_len=seq_num, padding=False)
        X_lps = X_lps[:, :, :, np.newaxis]

        X_lps_list.append(X_lps)
        X_info_list.extend(X_info)
        processed_classes.append(label)

        print(f"Successfully processed {X_lps.shape[0]} samples for class {label}")

    except Exception as e:
        print(f"Error processing class {label}: {e}")

# Verify data loading
if not X_lps_list:
    raise ValueError("No data was loaded for any class!")

print("\nData summary:")
for i, x_lps in enumerate(X_lps_list):
    print(f"Class {processed_classes[i]}: {x_lps.shape[0]} samples")

# Update labels to match processed classes
num_classes = len(processed_classes)
X_lps = np.concatenate(X_lps_list, axis=0)
X_label = np.hstack([np.ones(len(X_lps_list[i])) * processed_classes[i]
                     for i in range(len(X_lps_list))])
X_label = to_categorical(X_label, num_classes=10)  # Keep 10 classes for consistency

# First split the data without augmentation
X_train, X_test, y_train, y_test = train_test_split(X_lps, X_label, test_size=0.2, random_state=42, stratify=X_label)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Define augmentation functions
def add_noise(data, noise_factor=0.005):
    noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return data + noise_factor * noise

def time_shift(data, shift_range=2):
    shift = np.random.randint(-shift_range, shift_range)
    return np.roll(data, shift, axis=1)

def frequency_mask(data, max_mask_size=8):
    batch, height, width, channels = data.shape
    mask_size = np.random.randint(1, max_mask_size)
    mask_start = np.random.randint(0, height - mask_size)

    augmented = data.copy()
    augmented[:, mask_start:mask_start+mask_size, :, :] = 0
    return augmented

# 修改資料增強部分
X_train_final = [X_train]  # 先放入原始訓練數據
y_train_final = [y_train]

# 逐步增強並即時合併
X_train_final.append(add_noise(X_train))
y_train_final.append(y_train)

X_train_final.append(time_shift(X_train))
y_train_final.append(y_train)

# 立即合併並釋放記憶體
X_train = np.concatenate(X_train_final, axis=0)
y_train = np.concatenate(y_train_final, axis=0)
del X_train_final, y_train_final
gc.collect()

# Shuffle the augmented training data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"Training data shape after augmentation: {X_train.shape}")
print(f"Training labels shape after augmentation: {y_train.shape}")

# Define the model
Inputs = Input(shape=(None, seq_num, 1), name="input")

filters = [32, 64, 96, 128]
feature_scale = 4
filters = [int(x / feature_scale) for x in filters]  # Adjust filters based on feature_scale

# First convolutional layer (using dilation_rate=2)
x = Conv2D(filters=filters[0], kernel_size=(5, 5), strides=(1, 1), dilation_rate=2, activation='relu', padding='same')(Inputs) #Changed padding to 'same'
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x) #Changed padding to 'same'

# Second convolutional layer (using dilation_rate=2)
x = Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), dilation_rate=2, activation='relu', padding='same')(x) #Changed padding to 'same'
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x) #Changed padding to 'same'

# Third convolutional layer (using dilation_rate=4)
x = Conv2D(filters=filters[2], kernel_size=(3, 3), strides=(1, 1), dilation_rate=4, activation='relu', padding='same')(x) #Changed padding to 'same'
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x) #Changed padding to 'same'

# Fourth convolutional layer (using dilation_rate=4)
x = Conv2D(filters=filters[3], kernel_size=(3, 3), strides=(1, 1), dilation_rate=4, activation='relu', padding='same')(x) #Changed padding to 'same'
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x) #Changed padding to 'same'

# Additional convolutional layer to increase complexity
x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x) #Changed padding to 'same'

# Global Average Pooling
x = GlobalAveragePooling2D()(x)

# Dropout layer to prevent overfitting
x = Dropout(0.5)(x)

# Output layer (outputting the number of classes, using softmax activation)
Outputs = Dense(10, activation='softmax')(x)  # Changed from 5 to 10 classes

# Build the model
model = Model(inputs=Inputs, outputs=Outputs)

# Compile the model
optimizer = Adam(learning_rate=0.0001)  # Adjust learning rate if needed
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', AUC(name='auroc')])

# Display the model architecture
model.summary()

# Calculate class weights to handle class imbalance
from sklearn.utils import class_weight

y_integers = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights = dict(enumerate(class_weights))

# Early stopping callback
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
loss, accuracy, auroc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test AUROC: {auroc:.4f}")

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

labels_str = [
    'Normal-No Movement',
    'Normal-Dry Swallow',
    'Normal-Cracker',
    'Normal-Jelly',
    'Normal-Water Drinking',
    'Patient-No Movement',
    'Patient-Dry Swallow',
    'Patient-Cracker',
    'Patient-Jelly',
    'Patient-Water Drinking'
]

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels_str))

# Modified evaluation code
print("Evaluating on test files...")
test_accuracy, test_predictions = evaluate_files(model, test_files, test_labels)
print(f"File-level Test Accuracy: {test_accuracy:.4f}")

# Calculate confusion matrix and classification report on file level
cm = confusion_matrix(test_labels, test_predictions)
print("\nFile-level Confusion Matrix:")
print(cm)

print("\nFile-level Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=labels_str))