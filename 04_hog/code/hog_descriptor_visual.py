"""
HOG Descriptor Visualization
Final HOG visualization and comparison images using real images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
os.makedirs(OUT_DIR, exist_ok=True)


def load_image():
    """Load the input image"""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    img = np.array(Image.open(image_path))
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)
    gray = gray / 255.0
    gray = np.power(np.clip(gray, 1e-8, 1), 0.5)
    return gray, img


def compute_hog_histograms(gray, cell_size=8, num_bins=9):
    """Compute HOG cell histograms"""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi % 180
    
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    bin_width = 180.0 / num_bins
    
    histograms = np.zeros((cells_y, cells_x, num_bins))
    for cy in range(cells_y):
        for cx in range(cells_x):
            y_start = cy * cell_size
            x_start = cx * cell_size
            cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
            cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
            
            for py in range(cell_size):
                for px in range(cell_size):
                    mag = cell_mag[py, px]
                    angle = cell_dir[py, px]
                    bin_idx = angle / bin_width
                    lower_bin = int(bin_idx) % num_bins
                    upper_bin = (lower_bin + 1) % num_bins
                    upper_weight = bin_idx - int(bin_idx)
                    lower_weight = 1 - upper_weight
                    histograms[cy, cx, lower_bin] += mag * lower_weight
                    histograms[cy, cx, upper_bin] += mag * upper_weight
    
    return histograms


def create_hog_visualization_explained():
    """Create detailed explanation of HOG visualization with real image"""
    gray, original_rgb = load_image()
    histograms = compute_hog_histograms(gray)
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel 1: Original image
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title(f'Original Image ({gray.shape[1]}×{gray.shape[0]})', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Image with cell grid
    ax = axes[0, 1]
    ax.imshow(gray, cmap='gray')
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.3, alpha=0.7)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.3, alpha=0.7)
    ax.set_title(f'Cell Grid ({cells_x}×{cells_y} cells)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 3: Sample histogram from center
    ax = axes[0, 2]
    cy, cx = cells_y // 2, cells_x // 2
    hist = histograms[cy, cx]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax.bar(range(9), hist, color=colors, edgecolor='black')
    ax.set_xlabel('Orientation Bin (20° each)', fontsize=10)
    ax.set_ylabel('Magnitude Sum', fontsize=10)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=9)
    ax.set_title(f'Histogram for Center Cell ({cx},{cy})', fontsize=12, fontweight='bold')
    
    # Panel 4: Single cell HOG lines
    ax = axes[1, 0]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_facecolor('black')
    
    max_hist = hist.max() if hist.max() > 0 else 1
    for bin_idx in range(9):
        if hist[bin_idx] > max_hist * 0.1:
            angle = bin_idx * 20 + 10
            scale = (hist[bin_idx] / max_hist) * 0.8
            dx = scale * np.cos(np.radians(angle))
            dy = scale * np.sin(np.radians(angle))
            linewidth = 2 + scale * 3
            ax.plot([-dx, dx], [-dy, dy], color='white', linewidth=linewidth, alpha=0.8)
    
    ax.set_title('Single Cell HOG Visualization\n(line length ∝ magnitude)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Panel 5: Full HOG visualization
    ax = axes[1, 1]
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.15:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.85
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=1, alpha=0.8)
    
    ax.set_title('Complete HOG Visualization\n(edge orientations per cell)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Panel 6: Side by side comparison
    ax = axes[1, 2]
    ax.set_xlim(0, gray.shape[1] * 2 + 10)
    ax.set_ylim(gray.shape[0], 0)
    
    ax.imshow(gray, cmap='gray', extent=[0, gray.shape[1], gray.shape[0], 0])
    
    offset_x = gray.shape[1] + 10
    ax.axvspan(offset_x, offset_x + gray.shape[1], facecolor='black')
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = offset_x + cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.15:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.85
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=0.8, alpha=0.8)
    
    ax.set_title('Original Image vs HOG Features', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('Understanding the HOG Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_visualization_explained.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_visualization_explained.png")


def create_descriptor_structure():
    """Visualize the final descriptor structure"""
    gray, _ = load_image()
    h, w = gray.shape
    cell_size = 8
    cells_x = w // cell_size
    cells_y = h // cell_size
    blocks_x = cells_x - 1
    blocks_y = cells_y - 1
    total_blocks = blocks_x * blocks_y
    total_dim = total_blocks * 36
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    
    ax.text(8, 7.5, f'HOG Descriptor Structure ({total_dim:,}-D Vector)', fontsize=16, 
            ha='center', fontweight='bold')
    
    # Draw blocks
    block_width = 0.12
    displayed_blocks = min(40, total_blocks)
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    start_x = 1
    np.random.seed(42)
    for block_idx in range(displayed_blocks):
        color = colors[block_idx % 20]
        for i in range(36):
            height = np.random.uniform(0.3, 1.5)
            x = start_x + block_idx * block_width * 1.1 + (i / 36) * block_width
            rect = Rectangle((x, 4), block_width/40, height, facecolor=color, edgecolor='none')
            ax.add_patch(rect)
    
    ax.text(start_x + displayed_blocks * block_width * 1.1 + 0.5, 4.5, '...', fontsize=20, ha='center')
    
    # Labels
    ax.plot([start_x, start_x + block_width], [3.5, 3.5], 'k-', linewidth=2)
    ax.plot([start_x, start_x], [3.4, 3.6], 'k-', linewidth=2)
    ax.plot([start_x + block_width, start_x + block_width], [3.4, 3.6], 'k-', linewidth=2)
    ax.text(start_x + block_width/2, 3.0, 'Block 0\n(36 values)', fontsize=9, ha='center')
    
    # Structure breakdown
    structure_text = f"""
    Structure Breakdown:
    ━━━━━━━━━━━━━━━━━━
    • {blocks_x} × {blocks_y} = {total_blocks} blocks
    • Each block: 2×2 cells = 4 cells  
    • Each cell: 9-bin histogram
    • Values per block: 4 × 9 = 36
    
    Total: {total_blocks} × 36 = {total_dim:,} dimensions
    """
    ax.text(12, 3.5, structure_text, fontsize=10, family='monospace', va='center',
            bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    usage_text = """
    Usage:
    ─────
    • Input to SVM classifier
    • Compare with templates
    • Sliding window detection
    """
    ax.text(12, 1.5, usage_text, fontsize=10, family='monospace', va='center',
            bbox=dict(facecolor='lightgreen', edgecolor='green', linewidth=2))
    
    plt.savefig(os.path.join(OUT_DIR, 'hog_descriptor_structure.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_descriptor_structure.png")


def create_hog_vs_sift_comparison():
    """Create comparison between HOG and SIFT"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # HOG characteristics
    ax = axes[0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    ax.text(5, 9.2, 'HOG', fontsize=18, ha='center', fontweight='bold', color='blue')
    ax.add_patch(Rectangle((0.5, 0.5), 9, 8.2, facecolor='lightblue', edgecolor='blue', linewidth=3, alpha=0.3))
    
    hog_text = """
    Purpose:
    • Object detection (pedestrians, cars)
    • Scene classification

    Descriptor:
    • Dense descriptor (whole window)
    • Fixed size (e.g., 3780-D for 64×128)
    • Captures shape via edge orientations

    Key Features:
    ✓ Unsigned gradients (0°-180°)
    ✓ Cell-based histograms (8×8)
    ✓ Block normalization (2×2 cells)
    ✓ 50% block overlap

    Invariance:
    ✓ Illumination (via normalization)
    ✗ Scale (needs multi-scale search)
    ✗ Rotation (orientation-sensitive)

    Use Case:
    → Sliding window detection
    → "Is there a person here?"
    """
    ax.text(5, 4.5, hog_text, fontsize=10, ha='center', va='center', family='monospace')
    
    # SIFT characteristics
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    ax.text(5, 9.2, 'SIFT', fontsize=18, ha='center', fontweight='bold', color='green')
    ax.add_patch(Rectangle((0.5, 0.5), 9, 8.2, facecolor='lightgreen', edgecolor='green', linewidth=3, alpha=0.3))
    
    sift_text = """
    Purpose:
    • Feature matching
    • Object recognition
    • Image stitching

    Descriptor:
    • Sparse keypoints
    • 128-D per keypoint
    • Variable # of keypoints

    Key Features:
    ✓ Scale-space extrema detection
    ✓ Sub-pixel keypoint refinement
    ✓ Orientation assignment
    ✓ 4×4 spatial bins, 8 orientations

    Invariance:
    ✓ Illumination (via normalization)
    ✓ Scale (scale-space pyramid)
    ✓ Rotation (orientation normalized)

    Use Case:
    → Match keypoints between images
    → "Find this object in scene"
    """
    ax.text(5, 4.5, sift_text, fontsize=10, ha='center', va='center', family='monospace')
    
    plt.suptitle('HOG vs SIFT: Different Tools for Different Tasks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_vs_sift.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_vs_sift.png")


def create_pedestrian_detection_example():
    """Create pedestrian detection example visualization using real image"""
    gray, original_rgb = load_image()
    histograms = compute_hog_histograms(gray)
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Panel 1: Original image
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Input Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Sliding window concept
    ax = axes[1]
    ax.imshow(original_rgb)
    
    # Draw multiple detection windows
    window_positions = [
        (50, 50, 64, 128, 'red', 0.3),
        (150, 100, 64, 128, 'orange', 0.4),
        (250, 80, 64, 128, 'yellow', 0.5),
        (350, 150, 64, 128, 'lime', 0.8),
    ]
    
    for x, y, w, h, color, score in window_positions:
        if x + w <= gray.shape[1] and y + h <= gray.shape[0]:
            rect = Rectangle((x, y), w, h, facecolor='none', edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w/2, y - 5, f'{score:.1f}', color=color, fontsize=9, ha='center', fontweight='bold')
    
    ax.set_title('Sliding Window Detection\n(64×128 windows with scores)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 3: HOG visualization
    ax = axes[2]
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.15:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.85
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=0.8, alpha=0.8)
    
    ax.set_title('HOG Features\n(used for SVM classification)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG + SVM Object Detection Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_pedestrian_example.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_pedestrian_example.png")


if __name__ == "__main__":
    print("Generating HOG Descriptor Visualizations...")
    print("-" * 50)
    create_hog_visualization_explained()
    create_descriptor_structure()
    create_hog_vs_sift_comparison()
    create_pedestrian_detection_example()
    print("-" * 50)
    print("Done!")
