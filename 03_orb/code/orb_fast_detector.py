"""
FAST Corner Detection - Detailed Implementation
Features from Accelerated Segment Test
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')


# Bresenham circle offsets (16 pixels around center)
CIRCLE_OFFSETS = [
    (0, -3),   # 1: North
    (1, -3),   # 2
    (2, -2),   # 3
    (3, -1),   # 4
    (3, 0),    # 5: East
    (3, 1),    # 6
    (2, 2),    # 7
    (1, 3),    # 8
    (0, 3),    # 9: South
    (-1, 3),   # 10
    (-2, 2),   # 11
    (-3, 1),   # 12
    (-3, 0),   # 13: West
    (-3, -1),  # 14
    (-2, -2),  # 15
    (-1, -3),  # 16
]


def visualize_bresenham_circle():
    """
    Visualize the 16-pixel Bresenham circle used in FAST.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Circle diagram
    ax = axes[0]
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw grid
    for i in range(-4, 5):
        ax.axhline(y=i, color='lightgray', linewidth=0.5)
        ax.axvline(x=i, color='lightgray', linewidth=0.5)
    
    # Draw center pixel
    center = plt.Rectangle((-0.5, -0.5), 1, 1, fill=True, facecolor='red', edgecolor='black', linewidth=2)
    ax.add_patch(center)
    ax.text(0, 0, 'p', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Draw circle pixels
    colors = plt.cm.hsv(np.linspace(0, 1, 16))
    for i, (dx, dy) in enumerate(CIRCLE_OFFSETS):
        rect = plt.Rectangle((dx - 0.5, -dy - 0.5), 1, 1, fill=True, 
                            facecolor=colors[i], edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        ax.text(dx, -dy, str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw actual circle
    circle = Circle((0, 0), 3, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    
    ax.set_title('FAST: 16-Pixel Bresenham Circle\n(Radius = 3)', fontsize=14, fontweight='bold')
    ax.set_xlabel('x offset')
    ax.set_ylabel('y offset')
    
    # Right: High-speed test explanation
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax2.add_patch(title_box)
    ax2.text(5, 9.1, 'High-Speed Test (Optimization)', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    # Content
    content = """
Test pixels 1, 5, 9, 13 first (N, E, S, W):

    ○ ← 1 (North)
    
13 → ○   ○ ← 5 (East)
(West)
    ○ ← 9 (South)

If at least 3 of these 4 are:
  • ALL brighter than p + t, OR
  • ALL darker than p - t

Then → Continue with full 16-pixel test
Else → NOT a corner (skip!)

This rejects ~80% of non-corners quickly!
"""
    ax2.text(0.5, 4.5, content, fontsize=11, va='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_fast_circle.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_fast_circle.png")


def visualize_fast_test_example():
    """
    Visualize FAST corner test with example pixel values.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Example 1: IS a corner (many darker pixels)
    ax1 = axes[0]
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_aspect('equal')
    
    center_val = 150
    threshold = 20
    
    # Circle pixel values (darker = potential corner)
    circle_vals = [80, 75, 70, 72, 130, 135, 140, 138, 135, 130, 85, 78, 75, 72, 70, 82]
    
    # Draw center
    center = plt.Rectangle((-0.5, -0.5), 1, 1, fill=True, facecolor='gray', edgecolor='black', linewidth=2)
    ax1.add_patch(center)
    ax1.text(0, 0, f'{center_val}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw circle pixels with colors
    darker_count = 0
    brighter_count = 0
    for i, (dx, dy) in enumerate(CIRCLE_OFFSETS):
        val = circle_vals[i]
        if val < center_val - threshold:
            color = 'darkblue'  # Darker
            darker_count += 1
        elif val > center_val + threshold:
            color = 'darkred'  # Brighter
            brighter_count += 1
        else:
            color = 'lightgray'  # Similar
        
        rect = plt.Rectangle((dx - 0.5, -dy - 0.5), 1, 1, fill=True, 
                            facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(dx, -dy, f'{val}', ha='center', va='center', fontsize=8, 
                color='white' if color in ['darkblue', 'darkred'] else 'black')
    
    ax1.set_title(f'Example 1: IS a Corner\nCenter={center_val}, t={threshold}\n'
                  f'Darker (blue): {darker_count}, Brighter (red): {brighter_count}',
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Example 2: NOT a corner (mixed)
    ax2 = axes[1]
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_aspect('equal')
    
    # Mixed values
    circle_vals2 = [120, 180, 145, 160, 140, 175, 155, 125, 165, 135, 180, 145, 130, 175, 150, 140]
    
    center2 = plt.Rectangle((-0.5, -0.5), 1, 1, fill=True, facecolor='gray', edgecolor='black', linewidth=2)
    ax2.add_patch(center2)
    ax2.text(0, 0, f'{center_val}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    for i, (dx, dy) in enumerate(CIRCLE_OFFSETS):
        val = circle_vals2[i]
        if val < center_val - threshold:
            color = 'darkblue'
        elif val > center_val + threshold:
            color = 'darkred'
        else:
            color = 'lightgray'
        
        rect = plt.Rectangle((dx - 0.5, -dy - 0.5), 1, 1, fill=True, 
                            facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(dx, -dy, f'{val}', ha='center', va='center', fontsize=8,
                color='white' if color in ['darkblue', 'darkred'] else 'black')
    
    ax2.set_title(f'Example 2: NOT a Corner\nCenter={center_val}, t={threshold}\n'
                  f'(No 9+ contiguous of same type)',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Legend
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    legend_text = """
FAST-9 Corner Test Algorithm
═══════════════════════════════════

Given:
  Iₚ = center pixel intensity
  t  = threshold (typically 20)

For each of 16 circle pixels (Iₐ):

  If Iₐ > Iₚ + t  →  BRIGHTER (B)
  If Iₐ < Iₚ - t  →  DARKER (D)
  Otherwise       →  SIMILAR (S)


CORNER detected if:

  9+ CONTIGUOUS pixels are B
       OR
  9+ CONTIGUOUS pixels are D


Color Legend:
  ███  Darker  (Iₐ < Iₚ - t)
  ███  Brighter (Iₐ > Iₚ + t)
  ███  Similar (within threshold)
"""
    ax3.text(0.1, 5, legend_text, fontsize=10, va='center', family='monospace')
    
    # Color boxes
    ax3.add_patch(plt.Rectangle((1, 1.5), 0.5, 0.3, facecolor='darkblue'))
    ax3.add_patch(plt.Rectangle((1, 1.0), 0.5, 0.3, facecolor='darkred'))
    ax3.add_patch(plt.Rectangle((1, 0.5), 0.5, 0.3, facecolor='lightgray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_fast_example.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_fast_example.png")


def visualize_contiguous_check():
    """
    Visualize how contiguous pixel checking works.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Test cases
    test_cases = [
        {
            'name': 'Case 1: Corner (9 contiguous darker)',
            'labels': ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'S', 'S', 'B', 'B', 'S', 'S', 'S'],
            'is_corner': True
        },
        {
            'name': 'Case 2: Corner (12 contiguous brighter)',
            'labels': ['S', 'S', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'S', 'S'],
            'is_corner': True
        },
        {
            'name': 'Case 3: NOT Corner (8 darker, 1 short)',
            'labels': ['D', 'D', 'D', 'D', 'S', 'D', 'D', 'D', 'D', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
            'is_corner': False
        },
        {
            'name': 'Case 4: Corner (wrap-around)',
            'labels': ['D', 'D', 'D', 'D', 'D', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'D', 'D', 'D', 'D'],
            'is_corner': True
        }
    ]
    
    colors_map = {'D': 'darkblue', 'B': 'darkred', 'S': 'lightgray'}
    
    for idx, case in enumerate(test_cases):
        ax = axes[idx // 2, idx % 2]
        
        # Draw circular arrangement
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw center
        center = Circle((0, 0), 0.15, facecolor='gray', edgecolor='black', linewidth=2)
        ax.add_patch(center)
        ax.text(0, 0, 'p', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Draw circle pixels
        radius = 1.2
        for i, label in enumerate(case['labels']):
            angle = 2 * np.pi * i / 16 - np.pi / 2  # Start from top
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            color = colors_map[label]
            circle = Circle((x, y), 0.2, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            ax.text(x, y, str(i+1), ha='center', va='center', fontsize=8, 
                   color='white' if label != 'S' else 'black', fontweight='bold')
        
        # Add label sequence
        sequence = ''.join(case['labels'])
        result_color = 'green' if case['is_corner'] else 'red'
        result_text = '✓ CORNER' if case['is_corner'] else '✗ NOT CORNER'
        
        ax.set_title(f"{case['name']}\n{result_text}", 
                    fontsize=11, fontweight='bold', color=result_color)
        
        # Show sequence below
        ax.text(0, -1.8, f"Sequence: {sequence}", ha='center', fontsize=9, family='monospace')
    
    plt.suptitle('FAST Contiguous Pixel Test (Need 9+ consecutive B or D)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_fast_contiguous.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_fast_contiguous.png")


def visualize_fast_on_image():
    """
    Apply FAST to the real input image and visualize.
    """
    # Load real input image
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if os.path.exists(image_path):
        img_rgb = np.array(Image.open(image_path))
        if len(img_rgb.shape) == 3:
            img_full = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0
        else:
            img_full = img_rgb / 255.0
        # Use a cropped portion for detailed visualization
        img = img_full[100:300, 100:400].copy()
    else:
        # Fallback to synthetic test image
        img = np.zeros((200, 300), dtype=np.float64)
        img[:, :] = 0.5
        img[30:100, 30:120] = 0.9
        for y in range(20, 80):
            for x in range(150, 150 + (y - 20)):
                if x < 300:
                    img[y, x] = 0.2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Test Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Apply FAST
    threshold = 0.15
    corners = []
    h, w = img.shape
    
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            center = img[y, x]
            upper = center + threshold
            lower = center - threshold
            
            # Get circle pixels
            circle_vals = []
            for dx, dy in CIRCLE_OFFSETS:
                circle_vals.append(img[y + dy, x + dx])
            
            # Label pixels
            labels = []
            for val in circle_vals:
                if val > upper:
                    labels.append('B')
                elif val < lower:
                    labels.append('D')
                else:
                    labels.append('S')
            
            # Check for 9+ contiguous
            labels_ext = labels + labels
            max_b = max_d = 0
            cnt_b = cnt_d = 0
            
            for l in labels_ext:
                if l == 'B':
                    cnt_b += 1
                    cnt_d = 0
                    max_b = max(max_b, cnt_b)
                elif l == 'D':
                    cnt_d += 1
                    cnt_b = 0
                    max_d = max(max_d, cnt_d)
                else:
                    cnt_b = cnt_d = 0
            
            max_b = min(max_b, 16)
            max_d = min(max_d, 16)
            
            if max_b >= 9 or max_d >= 9:
                corners.append((x, y, max(max_b, max_d)))
    
    # Show detections
    ax2 = axes[1]
    ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
    for x, y, resp in corners:
        circle = Circle((x, y), 2, facecolor='none', edgecolor='lime', linewidth=1)
        ax2.add_patch(circle)
    ax2.set_title(f'FAST Corners Detected: {len(corners)}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Zoom on one corner
    ax3 = axes[2]
    
    # Pick a corner near the square
    corner_x, corner_y = 30, 30
    patch_size = 7
    
    ax3.imshow(img[corner_y-patch_size:corner_y+patch_size+1, 
                   corner_x-patch_size:corner_x+patch_size+1], 
               cmap='gray', vmin=0, vmax=1, extent=[-patch_size-0.5, patch_size+0.5, 
                                                    patch_size+0.5, -patch_size-0.5])
    
    # Draw circle
    for dx, dy in CIRCLE_OFFSETS:
        val = img[corner_y + dy, corner_x + dx]
        center = img[corner_y, corner_x]
        if val > center + threshold:
            color = 'red'
        elif val < center - threshold:
            color = 'blue'
        else:
            color = 'gray'
        circle = Circle((dx, dy), 0.4, facecolor=color, edgecolor='black', alpha=0.7)
        ax3.add_patch(circle)
    
    # Mark center
    ax3.plot(0, 0, 'y*', markersize=15)
    ax3.set_title(f'Zoomed Corner at ({corner_x}, {corner_y})\nBlue=Darker, Red=Brighter', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_fast_detection.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_fast_detection.png")


def main():
    """Generate all FAST detector visualizations."""
    print("=" * 60)
    print("FAST Corner Detector Visualizations")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("\n1. Generating Bresenham circle diagram...")
    visualize_bresenham_circle()
    
    print("\n2. Generating FAST test examples...")
    visualize_fast_test_example()
    
    print("\n3. Generating contiguous check visualization...")
    visualize_contiguous_check()
    
    print("\n4. Generating FAST on test image...")
    visualize_fast_on_image()
    
    print("\n" + "=" * 60)
    print("Done! Generated all FAST detector visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
