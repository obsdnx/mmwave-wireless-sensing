import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def read_mmwave_frame(file_path):
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(-1, 5)  # Reshape to (n_points, 5) for (x, y, z, d, I)

def visualize_mmwave_frame(frame, frame_number, save_path):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z, d, I = frame.T
    
    # Normalize intensity for better color distribution
    I_norm = (I - I.min()) / (I.max() - I.min())
    
    scatter = ax.scatter(x, y, z, c=I_norm, cmap='viridis', s=20, alpha=0.7)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'mmWave Frame {frame_number}', fontsize=14)
    
    # Set axis limits for better focus
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4)
    ax.set_zlim(-1, 3)
    
    # Add a color bar
    cbar = plt.colorbar(scatter, label='Normalized Intensity', pad=0.1)
    cbar.ax.tick_params(labelsize=10)
    
    # Improve grid and background
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    
    # Save the figure with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    # Optionally display the plot (comment out if not needed)
    plt.show()

if __name__ == "__main__":
    mmwave_folder = "DB_Coursework/S01/A01/mmWave"
    frame_to_visualize = "frame001.bin"  # Change this to visualize a different frame
    
    file_path = os.path.join(mmwave_folder, frame_to_visualize)
    frame = read_mmwave_frame(file_path)
    
    save_path = "mmwave_visualization_improved.png"
    visualize_mmwave_frame(frame, 0, save_path)
