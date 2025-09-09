import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'iloveyou', 'thanks'])
no_sequences = 30
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

print("Loading data...")
try:
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                
                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"File not found: {npy_path}")
                
                res = np.load(npy_path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

if np.isnan(X).any():
    print("WARNING: NaN values detected in data!")
    X = np.nan_to_num(X) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_mean = np.mean(X_train, axis=(0, 1))
X_std = np.std(X_train, axis=(0, 1)) + 1e-8

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

np.save('x_mean.npy', X_mean)
np.save('x_std.npy', X_std)
print("Normalization parameters saved.")

def augment_sequence(sequence, noise_factor=0.02):
    """Add slight noise and variations to sequences"""
    noise = np.random.normal(0, noise_factor, sequence.shape)
    scale = np.random.uniform(0.95, 1.05, sequence.shape[1])
    return sequence * scale + noise

X_train_augmented = [X_train[i] for i in range(len(X_train))]
y_train_augmented = [y_train[i] for i in range(len(y_train))]

for i in range(len(X_train)):
    for _ in range(2): 
        augmented_seq = augment_sequence(X_train[i])
        X_train_augmented.append(augmented_seq)
        y_train_augmented.append(y_train[i])

X_train = np.array(X_train_augmented)
y_train = np.array(y_train_augmented)

print(f"After augmentation - X_train shape: {X_train.shape}")
print("Building optimized model...")
model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='tanh', 
               input_shape=(sequence_length, 126),
               kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(LSTM(16, return_sequences=False, activation='tanh',
               kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(actions.shape[0], activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

model.summary()

log_dir = os.path.join('Logs')
tensorboard_callback = TensorBoard(log_dir=log_dir)
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', 
                                  save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, 
                             min_lr=0.00005, verbose=1)

callbacks = [tensorboard_callback, early_stopping, model_checkpoint, reduce_lr]

print("Starting enhanced training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=200,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

try:
    model.load_weights('best_model.h5')
    print("Loaded best model weights.")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Best model test accuracy: {test_accuracy:.4f}")
except:
    print("Using final model weights.")

model.save('best_model.keras')
print("Model saved as 'best_model.keras'")
