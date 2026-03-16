"""
SURF Step 5: Orientation Assignment Visualization
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_image():
    input_path = os.path.join(IMAGES_DIR, 'input_image.jpg')
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        return np.array(img)
    else:
        img = np.zeros((480, 640), dtype=np.uint8)
        img[100:200, 100:250] = 180
        img[250:350, 300:450] = 220
        Image.fromarray(img).save(input_path)
        return img


def get_sample_keypoints(img, n=50):
    """Generate sample keypoints for visualization"""
    H, W = img.shape
    np.random.seed(42)
    keypoints = []
    for _ in range(n):
        kp = {
            'x': np.random.randint(50, W - 50),
            'y': np.random.randint(50, H - 50),
            'scale': np.random.randint(1, 4),
            'orientation': np.random.uniform(-np.pi, np.pi)
        }
        keypoints.append(kp)
    return keypoints


def visualize_haar_wavelets():
    """Visualize Haar wavelet filters"""
    print("=" * 60)
    print("SURF Step 5: Orientation Assignment")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Haar X wavelet
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(10, 0)
    ax1.axis('off')
    
    # Left half: -1 (red), Right half: +1 (green)
    rect_left = patches.Rectangle((1, 2), 4, 6, facecolor='red', edgecolor='black', linewidth=2)
    rect_right = patches.Rectangle((5, 2), 4, 6, facecolor='green', edgecolor='black', linewidth=2)
    ax1.add_patch(rect_left)
    ax1.add_patch(rect_right)
    ax1.text(3, 5, '-1', fontsize=16, fontweight='bold', color='white', ha='center', va='center')
    ax1.text(7, 5, '+1', fontsize=16, fontweight='bold', color='white', ha='center', va='center')
    ax1.set_title('Haar Wavelet X (dx)\nHorizontal Gradient', fontsize=14, fontweight='bold')
    
    # Haar Y wavelet
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(10, 0)
    ax2.axis('off')
    
    # Top half: +1 (green), Bottom half: -1 (red)
    rect_top = patches.Rectangle((2, 1), 6, 4, facecolor='green', edgecolor='black', linewidth=2)
    rect_bottom = patches.Rectangle((2, 5), 6, 4, facecolor='red', edgecolor='black', linewidth=2)
    ax2.add_patch(rect_top)
    ax2.add_patch(rect_bottom)
    ax2.text(5, 3, '+1', fontsize=16, fontweight='bold', color='white', ha='center', va='center')
    ax2.text(5, 7, '-1', fontsize=16, fontweight='bold', color='white', ha='center', va='center')
    ax2.set_title('Haar Wavelet Y (dy)\nVertical Gradient', fontsize=14, fontweight='bold')
    
    # Formula and explanation
    ax3 = axes[2]
    ax3.axis('off')
    formula = """
Haar Wavelet Computation
════════════════════════════════

Using Integral Image:

dx = Sum(right) - Sum(left)
dy = Sum(bottom) - Sum(top)

With integral image II:
  dx = [II(D) - II(B) - II(C) + II(A)]_right
     - [II(D) - II(B) - II(C) + II(A)]_left

  → Only 8 lookups total!
  → O(1) computation!

Result:
  dx > 0: intensity increases left→right
  dx < 0: intensity decreases left→right
  dy > 0: intensity increases top→bottom
  dy < 0: intensity decreases top→bottom

Wavelet size = 4s (s = keypoint scale)
"""
    ax3.text(0.05, 0.5, formula, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Haar Wavelet Formula', fontsize=14, fontweight='bold')
    
    plt.suptitle('SURF Step 5: Haar Wavelets for Orientation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step5_haar.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step5_haar.png")


def visualize_sliding_window():
    """Visualize the 60-degree sliding window for orientation"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Circular region with sample points
    ax1 = axes[0]
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    
    # Draw circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
    
    # Sample points with gradient vectors
    np.random.seed(42)
    for _ in range(30):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.3, 0.9)
        x, y = r * np.cos(angle), r * np.sin(angle)
        
        # Random gradient direction
        grad_angle = np.random.uniform(0, 2*np.pi)
        grad_mag = np.random.uniform(0.05, 0.15)
        ax1.arrow(x, y, grad_mag*np.cos(grad_angle), grad_mag*np.sin(grad_angle),
                 head_width=0.03, head_length=0.02, fc='gray', ec='gray', alpha=0.7)
    
    # Highlight 60-degree window
    window_center = np.pi/4
    theta_win = np.linspace(window_center - np.pi/6, window_center + np.pi/6, 30)
    ax1.fill(np.concatenate([[0], 1.2*np.cos(theta_win), [0]]),
             np.concatenate([[0], 1.2*np.sin(theta_win), [0]]), 
             color='yellow', alpha=0.4, label='60° window')
    
    # Dominant direction
    ax1.arrow(0, 0, 0.9*np.cos(window_center), 0.9*np.sin(window_center),
             head_width=0.1, head_length=0.08, fc='red', ec='red', linewidth=3,
             label='Dominant orientation')
    
    ax1.plot(0, 0, 'ko', markersize=10)
    ax1.set_title('Circular Region (radius = 6s)\nwith Haar wavelet responses', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Sliding window process
    ax2 = axes[1]
    ax2.axis('off')
    process = """
60° Sliding Window Process
════════════════════════════════════════

STEP 1: Sample Haar Responses
────────────────────────────────────────
For each sample point in circular region:
  • Compute dx (horizontal gradient)
  • Compute dy (vertical gradient)
  • Weight by Gaussian(distance)

STEP 2: Rotate Window Around Circle
────────────────────────────────────────
For each window position (0° to 360°):
  
  sum_dx = Σ (dx × weight) in window
  sum_dy = Σ (dy × weight) in window
  
  magnitude = √(sum_dx² + sum_dy²)

STEP 3: Find Dominant Orientation
────────────────────────────────────────
Window with LARGEST magnitude wins!

  θ_dominant = atan2(sum_dy_max, sum_dx_max)

EXAMPLE:
────────────────────────────────────────
Window at 45°:
  sum_dx = 15.3
  sum_dy = 18.2
  magnitude = √(15.3² + 18.2²) = 23.8

Window at 135°:
  sum_dx = -8.5
  sum_dy = 12.1
  magnitude = √(8.5² + 12.1²) = 14.8

Winner: 45° window (magnitude 23.8)
  θ_dominant = atan2(18.2, 15.3) = 49.9°
"""
    ax2.text(0.02, 0.5, process, fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax2.set_title('Sliding Window Process', fontsize=14, fontweight='bold')
    
    plt.suptitle('SURF Step 5: 60° Sliding Window for Orientation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step5_sliding.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step5_sliding.png")


def visualize_keypoints_with_orientation():
    """Show keypoints with assigned orientations"""
    img = load_image()
    keypoints = get_sample_keypoints(img, n=40)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        ori = kp['orientation']
        scale = kp['scale']
        size = 5 + scale * 3
        
        # Draw keypoint circle
        circle = plt.Circle((x, y), size, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Draw orientation arrow
        dx = size * 1.8 * np.cos(ori)
        dy = size * 1.8 * np.sin(ori)
        ax.arrow(x, y, dx, dy, head_width=5, head_length=4, fc='yellow', ec='yellow', linewidth=1.5)
    
    ax.set_title(f'SURF Keypoints with Orientations\n{len(keypoints)} keypoints', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step5_orientation.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step5_orientation.png")


def visualize_orientation_details():
    """Detailed explanation of orientation assignment"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    details = """
SURF ORIENTATION ASSIGNMENT - Complete Details
══════════════════════════════════════════════════════════════════════════════

PURPOSE: Achieve ROTATION INVARIANCE
────────────────────────────────────────────────────────────────────────────
By assigning a dominant orientation to each keypoint, we can:
  • Rotate the descriptor region to align with this orientation
  • Match the same keypoint regardless of image rotation


STEP-BY-STEP PROCESS:
════════════════════════════════════════════════════════════════════════════

1. DEFINE CIRCULAR REGION
────────────────────────────────────────────────────────────────────────────
   • Center: keypoint location (x, y)
   • Radius: 6s (where s = keypoint scale)
   
   Example: For keypoint at scale s=2, radius = 6×2 = 12 pixels

2. COMPUTE HAAR WAVELET RESPONSES
────────────────────────────────────────────────────────────────────────────
   Sample points at regular intervals within the circle:
   
   For each sample point (px, py):
     • dx = Haar_X response (size = 4s)
     • dy = Haar_Y response (size = 4s)
     • weight = Gaussian(distance from center, σ = 2.5s)
     
   Result: A set of weighted (dx, dy) vectors

3. SLIDING WINDOW SUMMATION
────────────────────────────────────────────────────────────────────────────
   Use a 60° (π/3 radians) sliding window:
   
   For each window position θ from 0° to 360°:
     
     Select all (dx, dy) with angle in [θ - 30°, θ + 30°]
     
     sum_dx = Σ (dx × weight)
     sum_dy = Σ (dy × weight)
     
     magnitude = √(sum_dx² + sum_dy²)

4. SELECT DOMINANT ORIENTATION
────────────────────────────────────────────────────────────────────────────
   Find window with MAXIMUM magnitude
   
   θ_dominant = atan2(sum_dy_max, sum_dx_max)
   
   This is the keypoint's orientation!


WHY 60° WINDOW?
────────────────────────────────────────────────────────────────────────────
   • Too small: Sensitive to noise
   • Too large: May mix different gradient directions
   • 60°: Good balance - captures dominant direction robustly


COMPARISON WITH SIFT:
────────────────────────────────────────────────────────────────────────────
   SIFT: 36-bin histogram of gradient directions
   SURF: Haar wavelets + sliding window
   
   SURF is faster due to integral image computation!
"""
    ax.text(0.02, 0.5, details, fontsize=9.5, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Step 5: Orientation Assignment - Complete Details', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step5_details.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step5_details.png")


if __name__ == "__main__":
    visualize_haar_wavelets()
    visualize_sliding_window()
    visualize_keypoints_with_orientation()
    visualize_orientation_details()
    print("\nStep 5 images generated successfully!")
