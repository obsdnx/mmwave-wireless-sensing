#a more efficient
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, Dropout
from keras.layers import TimeDistributed, Reshape, GlobalAveragePooling1D

# Constants
PROCESSED_DIR = "/Users/anaisshakhidi/Desktop/cs345-cw/processed_data"
MODEL_DIR = "models3"
BATCH_SIZE = 16
EPOCHS = 3
N_KEYPOINTS = 17 * 3  # 17 joints with x, y, z coordinates

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(mfpc_files, gt_files):
    all_features, all_ground_truth = [], []
    
    for mfpc_file, gt_file in zip(mfpc_files, gt_files):
        try:
            mfpc_path = os.path.join(PROCESSED_DIR, mfpc_file)
            gt_path = os.path.join(PROCESSED_DIR, gt_file)

            if os.path.exists(mfpc_path) and os.path.exists(gt_path):
                mfpc_features = np.load(mfpc_path)
                ground_truth = np.load(gt_path).reshape(-1, N_KEYPOINTS)

                min_frames = min(mfpc_features.shape[0], ground_truth.shape[0])
                all_features.append(mfpc_features[:min_frames])
                all_ground_truth.append(ground_truth[:min_frames])
                print(f"Loaded: {mfpc_file} - Features: {mfpc_features.shape}, GT: {ground_truth.shape}")
        except Exception as e:
            print(f"Skipping {mfpc_file}/{gt_file}: {e}")

    return np.concatenate(all_features, axis=0), np.concatenate(all_ground_truth, axis=0)

def define_CNN(input_shape, n_keypoints):
    in_one = Input(shape=input_shape)
    
    x = Reshape((input_shape[0], input_shape[2], input_shape[3], input_shape[1]))(in_one)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Flatten())(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = GlobalAveragePooling1D()(x)
    out_layer = Dense(n_keypoints, activation='linear')(x)

    model = Model(inputs=in_one, outputs=out_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), 
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    return model

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.close()

def train_and_evaluate():
    train_mfpc_files = [f'train_mfpc_features_{i}.npy' for i in range(16)]
    train_gt_files = [f'train_ground_truth_{i}.npy' for i in range(16)]
    test_mfpc_files = [f'test_mfpc_features_{i}.npy' for i in range(10)]
    test_gt_files = [f'test_ground_truth_{i}.npy' for i in range(10)]

    print("Loading Training Data...")
    X_train, y_train = load_data(train_mfpc_files, train_gt_files)

    print("\nLoading Test Data...")
    X_test, y_test = load_data(test_mfpc_files, test_gt_files)

    print("\nData Shapes:")
    print(f"Training: {X_train.shape}, {y_train.shape}")
    print(f"Testing: {X_test.shape}, {y_test.shape}")

    model = define_CNN(X_train[0].shape, N_KEYPOINTS)
    model.summary()

    history = model.fit(X_train, y_train, validation_split=0.2, 
                        batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.save(os.path.join(MODEL_DIR, "trained_model.h5"))
    print("Model saved successfully!")

    plot_training_history(history)

    test_metrics = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_metrics[0]:.4f}")
    print(f"Test MAE: {test_metrics[1]:.4f}")
    print(f"Test RMSE: {test_metrics[2]:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
