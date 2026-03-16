"""
Visual diagram for SIFT Scale Factor Table and Coordinate Transformation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import os

def create_scale_factor_visual():
    """Create visual diagram showing octave scaling and coordinate transformation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ============================================
    # Panel 1: Octave Resolution Diagram
    # ============================================
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Octave Resolution Hierarchy', fontsize=14, fontweight='bold', pad=10)
    
    # Octave 0 - Full resolution
    rect0 = patches.FancyBboxPatch((1, 6), 4, 3, boxstyle="round,pad=0.05", 
                                    facecolor='#3498db', edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(rect0)
    ax1.text(3, 7.5, 'OCTAVE 0', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax1.text(3, 6.8, 'H × W', ha='center', va='center', fontsize=11, color='white')
    ax1.text(3, 6.2, '(Full Resolution)', ha='center', va='center', fontsize=9, color='white')
    
    # Octave 1 - Half resolution
    rect1 = patches.FancyBboxPatch((2, 3.2), 3, 2.2, boxstyle="round,pad=0.05",
                                    facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(rect1)
    ax1.text(3.5, 4.5, 'OCTAVE 1', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax1.text(3.5, 3.9, 'H/2 × W/2', ha='center', va='center', fontsize=10, color='white')
    ax1.text(3.5, 3.4, '(Half Resolution)', ha='center', va='center', fontsize=8, color='white')
    
    # Octave 2 - Quarter resolution
    rect2 = patches.FancyBboxPatch((2.5, 0.8), 2, 1.8, boxstyle="round,pad=0.05",
                                    facecolor='#27ae60', edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(rect2)
    ax1.text(3.5, 1.9, 'OCTAVE 2', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax1.text(3.5, 1.4, 'H/4 × W/4', ha='center', va='center', fontsize=9, color='white')
    ax1.text(3.5, 1.0, '(Quarter)', ha='center', va='center', fontsize=8, color='white')
    
    # Arrows showing downsampling
    ax1.annotate('', xy=(3.5, 5.5), xytext=(3, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.text(4.2, 5.6, '÷2', fontsize=10, fontweight='bold')
    
    ax1.annotate('', xy=(3.5, 2.7), xytext=(3.5, 3.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.text(4.0, 2.9, '÷2', fontsize=10, fontweight='bold')
    
    # Scale factors on right
    ax1.text(7, 7.5, 'Scale: ×1', fontsize=12, fontweight='bold', color='#3498db')
    ax1.text(7, 4.3, 'Scale: ×2', fontsize=12, fontweight='bold', color='#e74c3c')
    ax1.text(7, 1.7, 'Scale: ×4', fontsize=12, fontweight='bold', color='#27ae60')
    
    # ============================================
    # Panel 2: Coordinate Transformation Visual
    # ============================================
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Coordinate Transformation to Original Image', fontsize=14, fontweight='bold', pad=10)
    
    # Original image (target)
    orig_rect = patches.FancyBboxPatch((7, 1), 4.5, 8, boxstyle="round,pad=0.02",
                                        facecolor='#ecf0f1', edgecolor='black', linewidth=2)
    ax2.add_patch(orig_rect)
    ax2.text(9.25, 9.3, 'Original Image (H × W)', ha='center', fontsize=11, fontweight='bold')
    
    # Octave boxes and arrows
    # Octave 0
    o0_rect = patches.Rectangle((0.5, 6.5), 2.5, 2), 
    ax2.add_patch(patches.FancyBboxPatch((0.5, 6.5), 2.5, 2, boxstyle="round,pad=0.02",
                                          facecolor='#3498db', edgecolor='black', alpha=0.8))
    ax2.text(1.75, 7.5, 'Oct 0\n(50,50)', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Arrow from Oct 0 to original
    ax2.annotate('', xy=(7.2, 7.5), xytext=(3.2, 7.5),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
    ax2.text(5, 7.8, '×1', fontsize=11, fontweight='bold', color='#3498db')
    
    # Keypoint in original from Oct 0
    kp0 = Circle((8.5, 7.5), 0.15, facecolor='#3498db', edgecolor='black', linewidth=1.5)
    ax2.add_patch(kp0)
    ax2.text(8.8, 7.5, '(50,50)', fontsize=9, color='#3498db', fontweight='bold')
    
    # Octave 1
    ax2.add_patch(patches.FancyBboxPatch((0.8, 4), 2, 1.5, boxstyle="round,pad=0.02",
                                          facecolor='#e74c3c', edgecolor='black', alpha=0.8))
    ax2.text(1.8, 4.75, 'Oct 1\n(30,40)', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Arrow from Oct 1 to original
    ax2.annotate('', xy=(7.2, 4.75), xytext=(3.0, 4.75),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
    ax2.text(5, 5.05, '×2', fontsize=11, fontweight='bold', color='#e74c3c')
    
    # Keypoint in original from Oct 1
    kp1 = Circle((8.5, 4.5), 0.2, facecolor='#e74c3c', edgecolor='black', linewidth=1.5)
    ax2.add_patch(kp1)
    ax2.text(8.8, 4.5, '(60,80)', fontsize=9, color='#e74c3c', fontweight='bold')
    
    # Octave 2
    ax2.add_patch(patches.FancyBboxPatch((1.0, 1.5), 1.5, 1.2, boxstyle="round,pad=0.02",
                                          facecolor='#27ae60', edgecolor='black', alpha=0.8))
    ax2.text(1.75, 2.1, 'Oct 2\n(20,25)', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Arrow from Oct 2 to original
    ax2.annotate('', xy=(7.2, 2.1), xytext=(2.7, 2.1),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    ax2.text(5, 2.4, '×4', fontsize=11, fontweight='bold', color='#27ae60')
    
    # Keypoint in original from Oct 2
    kp2 = Circle((8.5, 2.2), 0.3, facecolor='#27ae60', edgecolor='black', linewidth=1.5)
    ax2.add_patch(kp2)
    ax2.text(8.9, 2.2, '(80,100)', fontsize=9, color='#27ae60', fontweight='bold')
    
    # ============================================
    # Panel 3: Scale Factor Table (Visual)
    # ============================================
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Scale Factor Table', fontsize=14, fontweight='bold', pad=10)
    
    # Table header
    headers = ['Octave', 'Resolution', 'Scale\nFactor', 'Transform']
    col_positions = [1, 3, 5.2, 7.5]
    
    # Header background
    header_rect = patches.FancyBboxPatch((0.3, 8), 9.4, 1.2, boxstyle="round,pad=0.02",
                                          facecolor='#34495e', edgecolor='black')
    ax3.add_patch(header_rect)
    
    for i, (header, pos) in enumerate(zip(headers, col_positions)):
        ax3.text(pos, 8.6, header, ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white')
    
    # Table rows
    rows = [
        ('0', 'H × W', '×1', '(x,y) → (x,y)'),
        ('1', 'H/2 × W/2', '×2', '(x,y) → (2x,2y)'),
        ('2', 'H/4 × W/4', '×4', '(x,y) → (4x,4y)'),
    ]
    colors = ['#3498db', '#e74c3c', '#27ae60']
    
    for idx, (row, color) in enumerate(zip(rows, colors)):
        y_pos = 6.5 - idx * 1.8
        
        # Row background
        row_rect = patches.FancyBboxPatch((0.3, y_pos - 0.6), 9.4, 1.4, boxstyle="round,pad=0.02",
                                           facecolor=color, edgecolor='black', alpha=0.3)
        ax3.add_patch(row_rect)
        
        for j, (val, pos) in enumerate(zip(row, col_positions)):
            ax3.text(pos, y_pos, val, ha='center', va='center', fontsize=11, 
                    fontweight='bold' if j == 0 else 'normal', color='black')
    
    # ============================================
    # Panel 4: Complete Flow Diagram
    # ============================================
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Complete Keypoint Combination Flow', fontsize=14, fontweight='bold', pad=10)
    
    # Flow boxes
    flow_data = [
        (0.5, 7.5, 'Octave 0\nDetect KP\nat (x₀,y₀)', '#3498db'),
        (0.5, 4.5, 'Octave 1\nDetect KP\nat (x₁,y₁)', '#e74c3c'),
        (0.5, 1.5, 'Octave 2\nDetect KP\nat (x₂,y₂)', '#27ae60'),
    ]
    
    for x, y, text, color in flow_data:
        box = patches.FancyBboxPatch((x, y), 2.5, 1.8, boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor='black', alpha=0.8)
        ax4.add_patch(box)
        ax4.text(x + 1.25, y + 0.9, text, ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold')
    
    # Scale boxes
    scale_data = [
        (4, 7.5, '×1\n(x₀,y₀)', '#3498db'),
        (4, 4.5, '×2\n(2x₁,2y₁)', '#e74c3c'),
        (4, 1.5, '×4\n(4x₂,4y₂)', '#27ae60'),
    ]
    
    for x, y, text, color in scale_data:
        box = patches.FancyBboxPatch((x, y), 2, 1.8, boxstyle="round,pad=0.05",
                                      facecolor='white', edgecolor=color, linewidth=2)
        ax4.add_patch(box)
        ax4.text(x + 1, y + 0.9, text, ha='center', va='center', 
                fontsize=10, color=color, fontweight='bold')
        
        # Arrow from detect to scale
        ax4.annotate('', xy=(x, y + 0.9), xytext=(3.2, y + 0.9),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Combine box
    combine_box = patches.FancyBboxPatch((7.5, 3.5), 3.5, 3, boxstyle="round,pad=0.05",
                                          facecolor='#9b59b6', edgecolor='black', linewidth=2, alpha=0.9)
    ax4.add_patch(combine_box)
    ax4.text(9.25, 5.5, 'COMBINE', ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
    ax4.text(9.25, 4.8, 'All Keypoints', ha='center', va='center', fontsize=10, color='white')
    ax4.text(9.25, 4.2, 'on Original', ha='center', va='center', fontsize=10, color='white')
    ax4.text(9.25, 3.7, 'Image', ha='center', va='center', fontsize=10, color='white')
    
    # Arrows to combine
    for y, color in [(8.4, '#3498db'), (5.4, '#e74c3c'), (2.4, '#27ae60')]:
        ax4.annotate('', xy=(7.5, 5), xytext=(6.2, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                   connectionstyle="arc3,rad=0.1"))
    
    # Formula at bottom
    ax4.text(6, 0.8, 'Final = KP₀ + KP₁(scaled) + KP₂(scaled)', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'sift_scale_factor_visual.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def create_coordinate_example():
    """Create a concrete numerical example of coordinate transformation"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Coordinate Transformation - Numerical Example', fontsize=16, fontweight='bold', pad=15)
    
    # Example scenario
    example_text = """
    Example: Image size 480 × 640 (H × W)
    """
    ax.text(7, 8.3, example_text.strip(), ha='center', fontsize=12, fontweight='bold')
    
    # Three octaves side by side
    octave_data = [
        {
            'name': 'Octave 0',
            'size': '480 × 640',
            'detected': '(150, 200)',
            'scale': '×1',
            'final': '(150, 200)',
            'color': '#3498db',
            'x_pos': 2
        },
        {
            'name': 'Octave 1', 
            'size': '240 × 320',
            'detected': '(60, 80)',
            'scale': '×2',
            'final': '(120, 160)',
            'color': '#e74c3c',
            'x_pos': 7
        },
        {
            'name': 'Octave 2',
            'size': '120 × 160',
            'detected': '(25, 30)',
            'scale': '×4',
            'final': '(100, 120)',
            'color': '#27ae60',
            'x_pos': 12
        }
    ]
    
    for data in octave_data:
        x = data['x_pos']
        color = data['color']
        
        # Octave name box
        name_box = patches.FancyBboxPatch((x - 1.8, 6.5), 3.6, 1, boxstyle="round,pad=0.05",
                                           facecolor=color, edgecolor='black', alpha=0.9)
        ax.add_patch(name_box)
        ax.text(x, 7, data['name'], ha='center', va='center', fontsize=12, 
               fontweight='bold', color='white')
        
        # Size
        ax.text(x, 6.1, f"Size: {data['size']}", ha='center', fontsize=10)
        
        # Detected keypoint
        detect_box = patches.FancyBboxPatch((x - 1.5, 4.5), 3, 1.2, boxstyle="round,pad=0.03",
                                             facecolor='#ecf0f1', edgecolor=color, linewidth=2)
        ax.add_patch(detect_box)
        ax.text(x, 5.3, 'Detected:', ha='center', fontsize=9, color='gray')
        ax.text(x, 4.8, data['detected'], ha='center', fontsize=11, fontweight='bold')
        
        # Arrow down
        ax.annotate('', xy=(x, 3.8), xytext=(x, 4.4),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        
        # Scale factor
        scale_box = patches.FancyBboxPatch((x - 0.6, 3.2), 1.2, 0.7, boxstyle="round,pad=0.03",
                                            facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(scale_box)
        ax.text(x, 3.55, data['scale'], ha='center', va='center', fontsize=12, 
               fontweight='bold', color='white')
        
        # Arrow down
        ax.annotate('', xy=(x, 2.3), xytext=(x, 3.1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        
        # Final coordinate
        final_box = patches.FancyBboxPatch((x - 1.5, 1.2), 3, 1.2, boxstyle="round,pad=0.03",
                                            facecolor=color, edgecolor='black', alpha=0.3)
        ax.add_patch(final_box)
        ax.text(x, 2, 'On Original:', ha='center', fontsize=9, color='gray')
        ax.text(x, 1.5, data['final'], ha='center', fontsize=11, fontweight='bold', color=color)
    
    # Calculation notes
    calc_notes = [
        '150×1 = 150\n200×1 = 200',
        '60×2 = 120\n80×2 = 160', 
        '25×4 = 100\n30×4 = 120'
    ]
    
    for i, (note, data) in enumerate(zip(calc_notes, octave_data)):
        ax.text(data['x_pos'], 0.5, note, ha='center', fontsize=9, 
               color='gray', style='italic')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'sift_coordinate_example.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


if __name__ == "__main__":
    create_scale_factor_visual()
    create_coordinate_example()
    print("\nDone! Generated visual diagrams for scale factor table.")
