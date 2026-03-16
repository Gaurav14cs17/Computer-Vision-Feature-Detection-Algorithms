"""
SIFT Algorithm Pipeline - Step by Step
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def gaussian_kernel(sigma, size=None):
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    center = size // 2
    kernel = np.zeros((size, size))
    for y in range(size):
        for x in range(size):
            dx, dy = x - center, y - center
            kernel[y, x] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(img, sigma):
    kernel = gaussian_kernel(sigma)
    return ndimage.convolve(img, kernel, mode='reflect')


def build_gaussian_pyramid(img, num_octaves=4, num_scales=5, sigma=1.6):
    print("[SIFT Step 1] Building Gaussian pyramid...")
    k = 2 ** (1.0 / (num_scales - 3))
    pyramid = []
    current_img = img.copy()
    
    for octave in range(num_octaves):
        octave_images = []
        for scale in range(num_scales):
            if scale == 0:
                blurred = gaussian_blur(current_img, sigma)
            else:
                sigma_total = sigma * (k ** scale)
                sigma_prev = sigma * (k ** (scale - 1))
                sigma_blur = np.sqrt(max(0.01, sigma_total**2 - sigma_prev**2))
                blurred = gaussian_blur(octave_images[-1], sigma_blur)
            octave_images.append(blurred)
        pyramid.append(octave_images)
        if octave < num_octaves - 1:
            base_idx = min(num_scales - 3, len(octave_images) - 1)
            current_img = octave_images[base_idx][::2, ::2]
    
    print(f"         Built {num_octaves} octaves x {num_scales} scales")
    return pyramid


def compute_dog_pyramid(gaussian_pyramid):
    print("[SIFT Step 2] Computing Difference of Gaussians...")
    dog_pyramid = []
    for octave_images in gaussian_pyramid:
        dog_octave = [octave_images[i + 1] - octave_images[i] for i in range(len(octave_images) - 1)]
        dog_pyramid.append(dog_octave)
    return dog_pyramid


def detect_keypoints(dog_pyramid, threshold=0.03):
    print("[SIFT Step 3] Detecting keypoints (DoG extrema)...")
    keypoints = []
    
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(dog_octave) - 1):
            prev_dog, curr_dog, next_dog = dog_octave[scale_idx - 1], dog_octave[scale_idx], dog_octave[scale_idx + 1]
            h, w = curr_dog.shape
            
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    val = curr_dog[y, x]
                    if abs(val) < threshold:
                        continue
                    
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                neighbors.extend([prev_dog[y + dy, x + dx], next_dog[y + dy, x + dx]])
                            else:
                                neighbors.extend([curr_dog[y + dy, x + dx], prev_dog[y + dy, x + dx], next_dog[y + dy, x + dx]])
                    
                    if all(val > n for n in neighbors) or all(val < n for n in neighbors):
                        scale_factor = 2 ** octave_idx
                        keypoints.append({'x': x * scale_factor, 'y': y * scale_factor, 'octave': octave_idx, 
                                         'scale': scale_idx, 'response': abs(val), 'local_x': x, 'local_y': y})
    
    print(f"         Found {len(keypoints)} keypoints")
    return keypoints


def refine_keypoints(keypoints, dog_pyramid, edge_threshold=10.0):
    print("[SIFT Step 4] Refining keypoints...")
    refined = []
    for kp in keypoints:
        dog = dog_pyramid[kp['octave']][kp['scale']]
        x, y = kp['local_x'], kp['local_y']
        h, w = dog.shape
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            continue
        dxx = dog[y, x + 1] + dog[y, x - 1] - 2 * dog[y, x]
        dyy = dog[y + 1, x] + dog[y - 1, x] - 2 * dog[y, x]
        dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1] - dog[y - 1, x + 1] + dog[y - 1, x - 1]) / 4
        det_h = dxx * dyy - dxy ** 2
        if det_h <= 0:
            continue
        if (dxx + dyy) ** 2 / det_h > (edge_threshold + 1) ** 2 / edge_threshold:
            continue
        refined.append(kp)
    print(f"         Refined to {len(refined)} keypoints")
    return refined


def compute_gradients(img):
    dx, dy = np.zeros_like(img), np.zeros_like(img)
    dx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    dy[1:-1, :] = img[2:, :] - img[:-2, :]
    return np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)


def assign_orientations(keypoints, gaussian_pyramid, num_bins=36):
    print("[SIFT Step 5] Assigning orientations...")
    oriented = []
    for kp in keypoints:
        img = gaussian_pyramid[kp['octave']][kp['scale']]
        h, w = img.shape
        x, y = kp['local_x'], kp['local_y']
        magnitude, orientation = compute_gradients(img)
        sigma = 1.5 * (1.6 * (2 ** (kp['scale'] / 3)))
        radius = int(3 * sigma)
        histogram = np.zeros(num_bins)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    weight = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    angle_deg = np.degrees(orientation[py, px]) % 360
                    histogram[int(angle_deg * num_bins / 360) % num_bins] += magnitude[py, px] * weight
        
        for _ in range(2):
            histogram = np.convolve(histogram, [1, 2, 1], mode='same') / 4
        
        max_val = histogram.max()
        for i in range(num_bins):
            if histogram[i] >= 0.8 * max_val:
                left, right = histogram[(i - 1) % num_bins], histogram[(i + 1) % num_bins]
                interp = 0.5 * (left - right) / (left - 2 * histogram[i] + right + 1e-6)
                new_kp = kp.copy()
                new_kp['orientation'] = np.radians((i + interp) * 360 / num_bins)
                oriented.append(new_kp)
    
    print(f"         {len(oriented)} keypoints with orientations")
    return oriented


def extract_descriptors(keypoints, gaussian_pyramid):
    print("[SIFT Step 6] Extracting SIFT descriptors (128-D)...")
    descriptors = []
    for kp in keypoints:
        img = gaussian_pyramid[kp['octave']][kp['scale']]
        h, w = img.shape
        x, y, orientation = kp['local_x'], kp['local_y'], kp['orientation']
        magnitude, angle = compute_gradients(img)
        cos_o, sin_o = np.cos(-orientation), np.sin(-orientation)
        descriptor = np.zeros(128)
        
        for i in range(4):
            for j in range(4):
                hist = np.zeros(8)
                for di in range(4):
                    for dj in range(4):
                        px, py = (i - 1.5) * 4 + di - 1.5, (j - 1.5) * 4 + dj - 1.5
                        rx, ry = int(x + px * cos_o - py * sin_o), int(y + px * sin_o + py * cos_o)
                        if 0 <= rx < w and 0 <= ry < h:
                            weight = np.exp(-(px**2 + py**2) / (2 * 64))
                            rel_angle = np.degrees(angle[ry, rx] - orientation) % 360
                            hist[int(rel_angle * 8 / 360) % 8] += magnitude[ry, rx] * weight
                descriptor[(i * 4 + j) * 8:(i * 4 + j) * 8 + 8] = hist
        
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = np.clip(descriptor / norm, 0, 0.2)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
        
        descriptors.append({'x': kp['x'], 'y': kp['y'], 'orientation': orientation, 'descriptor': descriptor})
    
    print(f"         Extracted {len(descriptors)} descriptors")
    return descriptors


def visualize_pyramid(pyramid, filename, title):
    num_octaves, num_scales = len(pyramid), len(pyramid[0])
    fig, axes = plt.subplots(num_octaves, min(num_scales, 5), figsize=(15, 3 * num_octaves))
    for o in range(num_octaves):
        for s in range(min(num_scales, 5)):
            ax = axes[o, s] if num_octaves > 1 else axes[s]
            ax.imshow(pyramid[o][s], cmap='gray')
            ax.set_title(f'Oct {o}, Scale {s}', fontsize=9)
            ax.axis('off')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_dog(dog_pyramid, filename):
    num_octaves, num_dogs = len(dog_pyramid), len(dog_pyramid[0])
    fig, axes = plt.subplots(num_octaves, min(num_dogs, 4), figsize=(12, 3 * num_octaves))
    for o in range(num_octaves):
        for d in range(min(num_dogs, 4)):
            ax = axes[o, d] if num_octaves > 1 else axes[d]
            ax.imshow(dog_pyramid[o][d], cmap='RdBu', vmin=-0.1, vmax=0.1)
            ax.set_title(f'DoG Oct {o}', fontsize=9)
            ax.axis('off')
    fig.suptitle('SIFT Step 2: Difference of Gaussians', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_keypoints(img, keypoints, filename, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray')
    for kp in keypoints[:300]:
        circle = plt.Circle((kp['x'], kp['y']), 5, color='lime', fill=False, linewidth=1)
        ax.add_patch(circle)
        if 'orientation' in kp:
            dx, dy = 12 * np.cos(kp['orientation']), 12 * np.sin(kp['orientation'])
            ax.arrow(kp['x'], kp['y'], dx, dy, head_width=3, head_length=2, fc='red', ec='red')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_descriptors(descriptors, filename):
    n = min(5, len(descriptors))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n))
    if n == 1:
        axes = [axes]
    for i, desc in enumerate(descriptors[:n]):
        axes[i].bar(range(128), desc['descriptor'], color='darkgreen', width=0.8)
        axes[i].set_xlim(-1, 128)
        if i == n - 1:
            axes[i].set_xlabel('Descriptor dimension (128-D)')
    fig.suptitle('SIFT Descriptors (128-D)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def run_sift_pipeline():
    print("=" * 70)
    print("SIFT ALGORITHM PIPELINE")
    print("=" * 70)
    
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if not os.path.exists(image_path):
        print("Creating test image...")
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:, :] = [100, 100, 100]
        img[50:100, 50:100] = [255, 255, 255]
        img[150:200, 200:280] = [200, 200, 200]
        Image.fromarray(img).save(image_path)
    
    img = np.array(Image.open(image_path))
    gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0 if len(img.shape) == 3 else img / 255.0
    
    gaussian_pyramid = build_gaussian_pyramid(gray)
    visualize_pyramid(gaussian_pyramid, "sift_step1_gaussian_pyramid.png", "SIFT Step 1: Gaussian Pyramid")
    
    dog_pyramid = compute_dog_pyramid(gaussian_pyramid)
    visualize_dog(dog_pyramid, "sift_step2_dog.png")
    
    keypoints = detect_keypoints(dog_pyramid)
    visualize_keypoints(gray, keypoints, "sift_step3_keypoints.png", f"SIFT Step 3: {len(keypoints)} keypoints")
    
    refined = refine_keypoints(keypoints, dog_pyramid)
    visualize_keypoints(gray, refined, "sift_step4_refined.png", f"SIFT Step 4: {len(refined)} refined")
    
    oriented = assign_orientations(refined, gaussian_pyramid)
    visualize_keypoints(gray, oriented, "sift_step5_orientation.png", f"SIFT Step 5: {len(oriented)} with orientation")
    
    descriptors = extract_descriptors(oriented, gaussian_pyramid)
    visualize_keypoints(gray, [{'x': d['x'], 'y': d['y'], 'orientation': d['orientation']} for d in descriptors],
                       "sift_step6_descriptors.png", f"SIFT Step 6: {len(descriptors)} descriptors")
    visualize_descriptors(descriptors, "sift_step6_descriptor_vectors.png")
    
    print("\n" + "=" * 70)
    print(f"COMPLETE: {len(descriptors)} keypoints, 128-D descriptors")
    print("=" * 70)
    return descriptors


if __name__ == "__main__":
    run_sift_pipeline()
