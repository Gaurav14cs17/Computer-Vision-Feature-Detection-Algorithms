"""
HOG Detailed Real Image Visualizations
Many more step-by-step images showing algorithm progress on real image
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import Normalize
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
    return gray, img


def gamma_correction(img, gamma=0.5):
    return np.power(np.clip(img, 1e-8, 1), gamma)


def compute_gradients(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi % 180
    return magnitude, direction, gx, gy


def compute_cell_histograms(magnitude, direction, cell_size=8, num_bins=9):
    h, w = magnitude.shape
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


# =============================================================================
# STEP 1 DETAILED REAL IMAGES
# =============================================================================

def create_step1_rgb_channels():
    """Show R, G, B channels separately"""
    gray, original_rgb = load_image()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title('Original RGB Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(original_rgb[:,:,0], cmap='Reds')
    ax.set_title('Red Channel (R)\nWeight: 0.299', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 0]
    ax.imshow(original_rgb[:,:,1], cmap='Greens')
    ax.set_title('Green Channel (G)\nWeight: 0.587', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(original_rgb[:,:,2], cmap='Blues')
    ax.set_title('Blue Channel (B)\nWeight: 0.114', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 1: RGB Channel Decomposition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step1_rgb_channels.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step1_rgb_channels.png")


def create_step1_grayscale_formula():
    """Show grayscale computation with formula overlay"""
    gray, original_rgb = load_image()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Input: RGB Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    weighted = 0.299 * original_rgb[:,:,0] + 0.587 * original_rgb[:,:,1] + 0.114 * original_rgb[:,:,2]
    ax.imshow(weighted, cmap='gray')
    ax.set_title('I = 0.299R + 0.587G + 0.114B', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[2]
    ax.imshow(gray, cmap='gray')
    ax.set_title('Output: Grayscale [0, 1]', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 1.1: Grayscale Conversion on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step1_grayscale_formula.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step1_grayscale_formula.png")


def create_step1_gamma_comparison():
    """Show multiple gamma values"""
    gray, _ = load_image()
    
    gammas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, gamma in enumerate(gammas):
        ax = axes[idx // 3, idx % 3]
        gamma_img = np.power(np.clip(gray, 1e-8, 1), gamma)
        ax.imshow(gamma_img, cmap='gray', vmin=0, vmax=1)
        if gamma == 0.5:
            ax.set_title(f'γ = {gamma} (HOG default) ✓', fontsize=11, fontweight='bold', color='green')
        else:
            ax.set_title(f'γ = {gamma}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('HOG Step 1.2: Gamma Correction Comparison on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step1_gamma_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step1_gamma_comparison.png")


def create_step1_histogram_equalization():
    """Show intensity histogram before/after gamma"""
    gray, _ = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.imshow(gray, cmap='gray')
    ax.set_title('Before Gamma', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(gray_gamma, cmap='gray')
    ax.set_title('After Gamma (γ=0.5)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 0]
    ax.hist(gray.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_title('Histogram Before', fontsize=12, fontweight='bold')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    ax.axvline(gray.mean(), color='red', linestyle='--', label=f'Mean: {gray.mean():.3f}')
    ax.legend()
    
    ax = axes[1, 1]
    ax.hist(gray_gamma.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.set_title('Histogram After (more spread out)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    ax.axvline(gray_gamma.mean(), color='red', linestyle='--', label=f'Mean: {gray_gamma.mean():.3f}')
    ax.legend()
    
    plt.suptitle('HOG Step 1.2: Gamma Effect on Intensity Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step1_histogram.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step1_histogram.png")


# =============================================================================
# STEP 2 DETAILED REAL IMAGES
# =============================================================================

def create_step2_gx_gy_combined():
    """Show Gx and Gy side by side with combined"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    ax.imshow(gray_gamma, cmap='gray')
    ax.set_title('Input (After Gamma)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(gx, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Gx: Horizontal Gradient\nDetects vertical edges', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 0]
    im = ax.imshow(gy, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Gy: Vertical Gradient\nDetects horizontal edges', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 1]
    combined = np.stack([np.abs(gx), np.abs(gy), np.zeros_like(gx)], axis=2)
    combined = combined / combined.max()
    ax.imshow(combined)
    ax.set_title('Combined: Red=|Gx|, Green=|Gy|\nYellow = both gradients', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 2: Gradient Components on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_gx_gy_combined.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_gx_gy_combined.png")


def create_step2_magnitude_thresholds():
    """Show magnitude at different thresholds"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, _, _, _ = compute_gradients(gray_gamma)
    
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, thresh in enumerate(thresholds):
        ax = axes[idx // 3, idx % 3]
        mask = magnitude > thresh
        display = np.zeros_like(original_rgb)
        display[mask] = original_rgb[mask]
        ax.imshow(display)
        count = np.sum(mask)
        pct = 100 * count / mask.size
        ax.set_title(f'Magnitude > {thresh}\n{count:,} pixels ({pct:.1f}%)', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('HOG Step 2: Edge Pixels at Different Magnitude Thresholds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_magnitude_thresholds.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_magnitude_thresholds.png")


def create_step2_direction_bins():
    """Show pixels colored by their direction bin"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    bin_names = ['0°-20°\n(→)', '20°-40°\n(↗)', '40°-60°\n(↗)', '60°-80°\n(↑)', '80°-100°\n(↑)', 
                 '100°-120°\n(↖)', '120°-140°\n(↖)', '140°-160°\n(←)', '160°-180°\n(←)']
    
    for bin_idx in range(9):
        ax = axes[bin_idx // 3, bin_idx % 3]
        
        bin_start = bin_idx * 20
        bin_end = (bin_idx + 1) * 20
        
        mask = (direction >= bin_start) & (direction < bin_end) & (magnitude > 0.03)
        
        display = np.ones((gray.shape[0], gray.shape[1], 3)) * 0.2
        display[mask] = colors[bin_idx][:3]
        
        ax.imshow(display)
        count = np.sum(mask)
        ax.set_title(f'Bin {bin_idx}: {bin_names[bin_idx]}\n{count:,} pixels', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('HOG Step 2: Pixels by Direction Bin (9 bins, 20° each)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_direction_bins.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_direction_bins.png")


def create_step2_edge_overlay():
    """Show edges overlaid on original image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(magnitude, cmap='hot')
    ax.set_title('Edge Magnitude (heat map)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[2]
    overlay = original_rgb.copy().astype(float) / 255
    edge_mask = magnitude > 0.05
    overlay[edge_mask, 0] = 1.0  # Red channel
    overlay[edge_mask, 1] *= 0.3
    overlay[edge_mask, 2] *= 0.3
    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_title('Edges Highlighted (red)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 2: Edge Detection on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_edge_overlay.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_edge_overlay.png")


def create_step2_gradient_field():
    """Show gradient field as quiver plot"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    ax.imshow(original_rgb, alpha=0.5)
    
    step = max(gray.shape[0] // 30, 10)
    y_coords = np.arange(step, gray.shape[0] - step, step)
    x_coords = np.arange(step, gray.shape[1] - step, step)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    U = gx[Y, X]
    V = gy[Y, X]
    M = magnitude[Y, X]
    
    mask = M > 0.02
    
    ax.quiver(X[mask], Y[mask], U[mask], V[mask], M[mask], 
              cmap='hot', scale=15, width=0.003, headwidth=4)
    
    ax.set_title('Gradient Field: Arrow = direction, Color = magnitude', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 2: Gradient Vector Field on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_gradient_field.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_gradient_field.png")


# =============================================================================
# STEP 3 DETAILED REAL IMAGES
# =============================================================================

def create_step3_cell_grid_detail():
    """Show cell grid with zoom on specific cells"""
    gray, original_rgb = load_image()
    
    cell_size = 8
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.3, alpha=0.5)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.3, alpha=0.5)
    
    zoom_cells = [(20, 30), (40, 50), (60, 40)]
    colors = ['red', 'blue', 'orange']
    for (cy, cx), color in zip(zoom_cells, colors):
        rect = Rectangle((cx*cell_size, cy*cell_size), cell_size, cell_size,
                         fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
    
    ax.set_title(f'Full Image: {cells_x}×{cells_y} = {cells_x*cells_y} cells', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    zoom_cy, zoom_cx = 40, 50
    zoom_region = original_rgb[zoom_cy*cell_size:(zoom_cy+3)*cell_size, 
                               zoom_cx*cell_size:(zoom_cx+3)*cell_size]
    ax.imshow(zoom_region, interpolation='nearest')
    for i in range(4):
        ax.axhline(y=i*cell_size, color='lime', linewidth=2)
        ax.axvline(x=i*cell_size, color='lime', linewidth=2)
    ax.set_title(f'Zoomed: 3×3 cells at ({zoom_cx}, {zoom_cy})\nEach cell = 8×8 = 64 pixels', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 3.1: Cell Grid on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_cell_grid_detail.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_cell_grid_detail.png")


def create_step3_single_cell_gradients():
    """Show gradients within a single cell"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    
    cell_size = 8
    cell_y, cell_x = 40, 50
    
    y_start = cell_y * cell_size
    x_start = cell_x * cell_size
    
    cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_gray = gray_gamma[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_gx = gx[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_gy = gy[y_start:y_start+cell_size, x_start:x_start+cell_size]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    ax = axes[0, 0]
    ax.imshow(cell_img, interpolation='nearest')
    ax.set_title(f'Cell ({cell_x}, {cell_y}): RGB', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(cell_gray, cmap='gray', interpolation='nearest')
    ax.set_title('Grayscale (γ corrected)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[0, 2]
    im = ax.imshow(cell_mag, cmap='hot', interpolation='nearest')
    ax.set_title('Magnitude M', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 0]
    im = ax.imshow(cell_gx, cmap='RdBu', vmin=-0.3, vmax=0.3, interpolation='nearest')
    ax.set_title('Gx (horizontal)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 1]
    im = ax.imshow(cell_gy, cmap='RdBu', vmin=-0.3, vmax=0.3, interpolation='nearest')
    ax.set_title('Gy (vertical)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 2]
    im = ax.imshow(cell_dir, cmap='hsv', vmin=0, vmax=180, interpolation='nearest')
    ax.set_title('Direction θ (0°-180°)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.suptitle(f'HOG Step 3.2: Single Cell Analysis - Cell ({cell_x}, {cell_y})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_single_cell_gradients.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_single_cell_gradients.png")


def create_step3_cell_histogram_real():
    """Show actual histogram for a real cell"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_y, cell_x = 40, 50
    cell_size = 8
    
    y_start = cell_y * cell_size
    x_start = cell_x * cell_size
    cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
    hist = histograms[cell_y, cell_x]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(cell_img, interpolation='nearest')
    ax.set_title(f'Cell ({cell_x}, {cell_y})', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    bars = ax.bar(range(9), hist, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°' for i in range(9)])
    ax.set_xlabel('Orientation Bin')
    ax.set_ylabel('Vote Magnitude')
    ax.set_title('9-Bin Histogram', fontsize=12, fontweight='bold')
    
    dominant_bin = np.argmax(hist)
    bars[dominant_bin].set_edgecolor('red')
    bars[dominant_bin].set_linewidth(3)
    
    ax = axes[2]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    max_val = hist.max()
    for bin_idx in range(9):
        angle = np.radians(bin_idx * 20 + 10)
        length = 0.3 + 1.0 * hist[bin_idx] / max_val
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        color = colors[bin_idx]
        ax.arrow(0, 0, dx, dy, head_width=0.08, head_length=0.05, fc=color, ec='black', linewidth=1)
    
    ax.add_patch(plt.Circle((0, 0), 0.3, fill=False, color='gray', linestyle='--'))
    ax.set_title('Histogram as Vectors\n(Length = vote magnitude)', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'HOG Step 3.3-3.4: Cell Histogram for Cell ({cell_x}, {cell_y})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_cell_histogram_real.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_cell_histogram_real.png")


def create_step3_multiple_cells_histograms():
    """Show histograms for multiple cells"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_positions = [(20, 20), (20, 60), (40, 40), (50, 30), (50, 70)]
    cell_size = 8
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for idx, (cy, cx) in enumerate(cell_positions):
        y_start = cy * cell_size
        x_start = cx * cell_size
        cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
        hist = histograms[cy, cx]
        
        ax = axes[0, idx]
        ax.imshow(cell_img, interpolation='nearest')
        ax.set_title(f'Cell ({cx}, {cy})', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        ax = axes[1, idx]
        colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
        ax.bar(range(9), hist, color=colors, edgecolor='black', width=0.8)
        ax.set_xticks([0, 4, 8])
        ax.set_xticklabels(['0°', '80°', '160°'], fontsize=8)
        dominant = np.argmax(hist)
        ax.set_title(f'Peak: {dominant*20}°-{(dominant+1)*20}°', fontsize=9)
    
    plt.suptitle('HOG Step 3.5: Histograms for Different Cells', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_multiple_cells_histograms.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_multiple_cells_histograms.png")


def create_step3_hog_overlay():
    """Show HOG visualization overlaid on original"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    ax.imshow(original_rgb, alpha=0.4)
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.25:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2.5)
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='red', linewidth=0.8, alpha=0.9)
    
    ax.set_title('HOG Features Overlaid on Original Image', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 3: Complete Cell Histograms Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_hog_overlay.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_hog_overlay.png")


# =============================================================================
# STEP 4 DETAILED REAL IMAGES
# =============================================================================

def create_step4_block_examples():
    """Show specific blocks with their 4 cells"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_size = 8
    block_positions = [(20, 30), (40, 50), (30, 60)]
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for row, (by, bx) in enumerate(block_positions):
        y_start = by * cell_size
        x_start = bx * cell_size
        
        block_img = original_rgb[y_start:y_start+cell_size*2, x_start:x_start+cell_size*2]
        ax = axes[row, 0]
        ax.imshow(block_img, interpolation='nearest')
        for i in range(3):
            ax.axhline(y=i*cell_size, color='lime', linewidth=2)
            ax.axvline(x=i*cell_size, color='lime', linewidth=2)
        ax.set_title(f'Block ({bx}, {by})\n2×2 cells', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        cell_names = ['(0,0)', '(1,0)', '(0,1)', '(1,1)']
        for idx, (dy, dx) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            cy, cx = by + dy, bx + dx
            hist = histograms[cy, cx]
            
            ax = axes[row, idx + 1]
            colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
            ax.bar(range(9), hist, color=colors, edgecolor='black', width=0.8)
            ax.set_title(f'Cell {cell_names[idx]}\n9 bins', fontsize=9, fontweight='bold')
            ax.set_xticks([])
    
    plt.suptitle('HOG Step 4.1: Block Structure - Each Block Contains 4 Cell Histograms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step4_block_examples.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step4_block_examples.png")


def create_step4_normalization_effect():
    """Show before/after normalization for a block"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    by, bx = 40, 50
    
    block_hists = []
    for dy in range(2):
        for dx in range(2):
            block_hists.append(histograms[by + dy, bx + dx])
    
    block_vector = np.concatenate(block_hists)
    
    norm = np.sqrt(np.sum(block_vector**2) + 1e-12)
    block_normalized = block_vector / norm
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    colors = np.tile(plt.cm.tab10(np.arange(4) / 4), (9, 1)).T.flatten().reshape(-1, 4)
    ax.bar(range(36), block_vector, color=colors[:36], edgecolor='black', width=0.8)
    ax.set_title(f'Before Normalization\nL2 Norm = {norm:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index (4 cells × 9 bins)')
    ax.set_ylabel('Value')
    
    ax = axes[1]
    ax.bar(range(36), block_normalized, color=colors[:36], edgecolor='black', width=0.8)
    ax.set_title(f'After L2 Normalization\nL2 Norm = 1.0', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Normalized Value')
    
    ax = axes[2]
    ax.axis('off')
    formula = """
    L2 Normalization Formula:
    
    v_normalized = v / √(||v||₂² + ε²)
    
    where:
    • v = 36-element block vector
    • ||v||₂ = √(Σᵢ vᵢ²) = L2 norm
    • ε = 1e-6 (numerical stability)
    
    Result:
    • ||v_normalized||₂ ≈ 1.0
    • Illumination invariant!
    """
    ax.text(0.1, 0.5, formula, fontsize=12, family='monospace', va='center',
            transform=ax.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange'))
    
    plt.suptitle(f'HOG Step 4.3: L2 Normalization Effect on Block ({bx}, {by})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step4_normalization_effect.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step4_normalization_effect.png")


def create_step4_final_descriptor_real():
    """Show final descriptor statistics"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    
    all_blocks = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = []
            for dy in range(2):
                for dx in range(2):
                    block.append(histograms[by + dy, bx + dx])
            block_vector = np.concatenate(block)
            norm = np.sqrt(np.sum(block_vector**2) + 1e-12)
            all_blocks.append(block_vector / norm)
    
    descriptor = np.concatenate(all_blocks)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title('Input Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.bar(range(min(500, len(descriptor))), descriptor[:500], color='darkblue', width=1)
    ax.set_title(f'Final Descriptor (first 500 of {len(descriptor):,})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Value')
    
    ax = axes[1, 0]
    ax.hist(descriptor, bins=50, color='green', edgecolor='black', alpha=0.7)
    ax.set_title('Descriptor Value Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.axvline(descriptor.mean(), color='red', linestyle='--', label=f'Mean: {descriptor.mean():.4f}')
    ax.legend()
    
    ax = axes[1, 1]
    ax.axis('off')
    stats = f"""
    ╔════════════════════════════════════════╗
    ║     HOG Descriptor Statistics          ║
    ╠════════════════════════════════════════╣
    ║                                        ║
    ║  Image Size:    {gray.shape[1]:4d} × {gray.shape[0]:<4d} pixels     ║
    ║  Cells:         {cells_x:4d} × {cells_y:<4d} = {cells_x*cells_y:<5d}      ║
    ║  Blocks:        {blocks_x:4d} × {blocks_y:<4d} = {blocks_x*blocks_y:<5d}      ║
    ║                                        ║
    ║  Descriptor Length: {len(descriptor):>7,d}           ║
    ║                                        ║
    ║  Value Statistics:                     ║
    ║    Min:   {descriptor.min():>10.6f}               ║
    ║    Max:   {descriptor.max():>10.6f}               ║
    ║    Mean:  {descriptor.mean():>10.6f}               ║
    ║    Std:   {descriptor.std():>10.6f}               ║
    ║                                        ║
    ╚════════════════════════════════════════╝
    """
    ax.text(0.1, 0.5, stats, fontsize=11, family='monospace', va='center',
            transform=ax.transAxes)
    
    plt.suptitle('HOG Step 4.5: Final Descriptor from Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step4_final_descriptor_real.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step4_final_descriptor_real.png")


if __name__ == "__main__":
    print("=" * 70)
    print("Generating HOG Detailed Real Image Visualizations")
    print("=" * 70)
    
    print("\n--- Step 1 Detailed ---")
    create_step1_rgb_channels()
    create_step1_grayscale_formula()
    create_step1_gamma_comparison()
    create_step1_histogram_equalization()
    
    print("\n--- Step 2 Detailed ---")
    create_step2_gx_gy_combined()
    create_step2_magnitude_thresholds()
    create_step2_direction_bins()
    create_step2_edge_overlay()
    create_step2_gradient_field()
    
    print("\n--- Step 3 Detailed ---")
    create_step3_cell_grid_detail()
    create_step3_single_cell_gradients()
    create_step3_cell_histogram_real()
    create_step3_multiple_cells_histograms()
    create_step3_hog_overlay()
    
    print("\n--- Step 4 Detailed ---")
    create_step4_block_examples()
    create_step4_normalization_effect()
    create_step4_final_descriptor_real()
    
    print("\n" + "=" * 70)
    print("Done! Generated 18 additional detailed real image visualizations.")
    print("=" * 70)
