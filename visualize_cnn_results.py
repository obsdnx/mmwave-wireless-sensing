import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cnn_model_simple


# Load the test frames, predictions, and ground truth
def load_test_data(base_dir):
    test_frames = np.load(os.path.join(base_dir, 'test', 'S10_frames.npy'))  # Adjust filename as needed
    test_labels = np.load(os.path.join(base_dir, 'test', 'S10_ground_truth.npy'))  # Adjust filename as needed
    predictions = np.load(os.path.join(base_dir, 'cnn_predictions', 'predictions.npy'))  # Ensure this path is correct
    return test_frames, test_labels, predictions

# Normalize data for better visualization
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Visualize results in a 3D plot
def visualize_results(test_frames, test_labels, predictions):
    # Normalize data for better visualization
    norm_test_frames = normalize_data(test_frames)
    norm_test_labels = normalize_data(test_labels)
    norm_predictions = normalize_data(predictions)

    fig = plt.figure(figsize=(18, 6))

    # Create a 3D subplot for input frames
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(norm_test_frames[:, 0], norm_test_frames[:, 1], norm_test_frames[:, 2], c='blue', label='Input Frames', alpha=0.5)
    ax1.set_title('Input Frames')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')

    # Create a 3D subplot for ground truth
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(norm_test_labels[:, 0], norm_test_labels[:, 1], norm_test_labels[:, 2], c='green', label='Ground Truth', alpha=0.5)
    ax2.set_title('Ground Truth')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')

    # Create a 3D subplot for CNN predictions
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(norm_predictions[:, 0], norm_predictions[:, 1], norm_predictions[:, 2], c='red', label='CNN Predictions', alpha=0.5)
    ax3.set_title('CNN Predictions')
    ax3.set_xlabel('X-axis')
    ax3.set_ylabel('Y-axis')
    ax3.set_zlabel('Z-axis')

    plt.legend()
    plt.tight_layout()
    plt.show()

# Main execution
base_dir = '/Users/anaisshakhidi/Desktop/cs345-cw/CNN_Data_Simple'
test_frames, test_labels, predictions = load_test_data(base_dir)
visualize_results(test_frames, test_labels, predictions)
