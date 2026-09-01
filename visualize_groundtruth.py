import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os


#Preamble: visualise the ground truth and visulalise without saving 

def load_npy_file(file_path):
    return np.load(file_path)

def process_npy_file(file_path):
    raw_data = load_npy_file(file_path)
    if 'ground_truth' in file_path:
        return process_ground_truth(raw_data, file_path)
    else:
        return process_mmwave(raw_data)

def process_ground_truth(data, file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    visualize_3d_joints(data, f'3D Joint Positions - {file_name}')
    animate_3d_joints(data, f'3D Joint Positions - {file_name}')
    return None, None

def process_mmwave(raw_data):
    if raw_data.ndim == 3:
        num_samples, num_channels, num_chirps = raw_data.shape
        virtual_data = np.zeros((num_samples, num_channels * num_chirps))
        for i in range(num_chirps):
            virtual_data[:, i * num_channels:(i + 1) * num_channels] = raw_data[:, :, i]
    else:
        virtual_data = raw_data

    rangedoppler = fftshift(fft2(virtual_data), axes=1)
    anglerange = np.fft.fft(virtual_data, axis=1)

    return rangedoppler, anglerange

def visualize_3d_joints(data, title):
    fig = plt.figure(figsize=(20, 16))
    
    # Calculate the overall min and max for consistent scaling
    overall_min = np.min(data)
    overall_max = np.max(data)
    
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        frame = i * (data.shape[0] // 4)
        x, y, z = data[frame, :, 0], data[frame, :, 2], -data[frame, :, 1]  # Plot as (x,z,-y)
        ax.scatter(x, y, z, c='b', marker='o', s=50, alpha=0.6, edgecolors='w')
        
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Z', fontsize=10)
        ax.set_zlabel('-Y', fontsize=10)
        ax.set_title(f'Frame {frame}', fontsize=12)
        ax.grid(True)
        
        # Set consistent axis limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        
        # Set aspect ratio to be equal
        ax.set_box_aspect((1, 1, 1))
        
        # Set consistent view for all subplots
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def animate_3d_joints(data, title, frame_step=5):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        x, y, z = data[frame * frame_step, :, 0], data[frame * frame_step, :, 2], -data[frame * frame_step, :, 1]  # Plot as (x,z,-y)
        ax.scatter(x, y, z, c='b', marker='o', s=50, alpha=0.6, edgecolors='w')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Z', fontsize=12)
        ax.set_zlabel('-Y', fontsize=12)
        ax.set_title(f'{title} - Frame {frame * frame_step}', fontsize=14)
        ax.grid(True)
        
        # Set consistent axis limits
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        
        # Set aspect ratio to be equal
        ax.set_box_aspect((1, 1, 1))
        
        ax.view_init(elev=20, azim=45)
    
    num_frames = data.shape[0] // frame_step
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)
    plt.show()

def plot_responses(rangedoppler, anglerange, file_name):
    if rangedoppler is not None and anglerange is not None:
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        
        # Range-Doppler Response
        im1 = axs[0, 0].imshow(20 * np.log10(np.abs(rangedoppler)), aspect='auto', cmap='viridis', origin='lower')
        axs[0, 0].set_xlabel('Doppler', fontsize=10)
        axs[0, 0].set_ylabel('Range', fontsize=10)
        axs[0, 0].set_title('Range-Doppler Response', fontsize=12)
        plt.colorbar(im1, ax=axs[0, 0], label='Magnitude (dB)')
        
        # Angle-Range Response
        im2 = axs[0, 1].imshow(20 * np.log10(np.abs(anglerange)), aspect='auto', cmap='plasma', origin='lower')
        axs[0, 1].set_xlabel('Angle', fontsize=10)
        axs[0, 1].set_ylabel('Range', fontsize=10)
        axs[0, 1].set_title('Angle-Range Response', fontsize=12)
        plt.colorbar(im2, ax=axs[0, 1], label='Magnitude (dB)')
        
        # Range-Doppler Response (zoomed)
        im3 = axs[1, 0].imshow(20 * np.log10(np.abs(rangedoppler[:rangedoppler.shape[0]//2, :])), aspect='auto', cmap='viridis', origin='lower')
        axs[1, 0].set_xlabel('Doppler', fontsize=10)
        axs[1, 0].set_ylabel('Range', fontsize=10)
        axs[1, 0].set_title('Range-Doppler Response (Zoomed)', fontsize=12)
        plt.colorbar(im3, ax=axs[1, 0], label='Magnitude (dB)')
        
        # Angle-Range Response (zoomed)
        im4 = axs[1, 1].imshow(20 * np.log10(np.abs(anglerange[:anglerange.shape[0]//2, :])), aspect='auto', cmap='plasma', origin='lower')
        axs[1, 1].set_xlabel('Angle', fontsize=10)
        axs[1, 1].set_ylabel('Range', fontsize=10)
        axs[1, 1].set_title('Angle-Range Response (Zoomed)', fontsize=12)
        plt.colorbar(im4, ax=axs[1, 1], label='Magnitude (dB)')
        
        plt.suptitle(f'mmWave Responses - {file_name}', fontsize=16)
        plt.tight_layout()
        plt.show()

# Main execution
dataset_path = 'DB_Coursework'

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(root, file)
            print(f"Processing: {file_path}")
            
            try:
                rangedoppler, anglerange = process_npy_file(file_path)
                plot_responses(rangedoppler, anglerange, os.path.splitext(file)[0])
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

print("Processing complete. All visualizations have been displayed.")
