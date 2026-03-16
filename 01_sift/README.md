# Understanding SIFT: Scale-Invariant Feature Transform

*A comprehensive guide to implementing SIFT from scratch*

---

The **Scale-Invariant Feature Transform (SIFT)** is one of the most influential algorithms in computer vision. Introduced by David Lowe in 2004, SIFT detects and describes local features in images that remain stable across changes in scale, rotation, and illumination.

This article walks through the complete SIFT pipeline, from mathematical foundations to practical implementation.

## Table of Contents

1. [Overview](#overview)
2. [Detection Phase](#detection-phase)
   - [Step 1: Gaussian Scale-Space Pyramid](#step-1-gaussian-scale-space-pyramid)
   - [Step 2: Difference of Gaussians](#step-2-difference-of-gaussians-dog)
   - [Step 3: Keypoint Detection](#step-3-keypoint-detection)
   - [Step 4: Keypoint Filtering & Refinement](#step-4-keypoint-filtering--refinement)
3. [Description Phase](#description-phase)
   - [Step 5: Orientation Assignment](#step-5-orientation-assignment)
   - [Step 6: Descriptor Extraction](#step-6-descriptor-extraction)
4. [Summary](#summary)

---

## Overview

SIFT operates in two main phases:

| Phase | Step | Description | Math |
|-------|------|-------------|------|
| Detection | 1 | Gaussian pyramid | `L(x,y,σ) = G(x,y,σ) * I(x,y)` |
| Detection | 2 | DoG | `D = L(kσ) - L(σ)` |
| Detection | 3 | Keypoint detection | 26-neighbor extrema |
| Detection | 4 | Refinement & Filtering | Taylor expansion + edge removal |
| Description | 5 | Orientation | 36-bin histogram |
| Description | 6 | Descriptor | 128-D |

### Project Structure

```
sift/
├── README.md                   ← Documentation
├── code/
│   ├── sift_pipeline.py        ← Main implementation
│   ├── sift_step3_pyramid.py   ← Detailed Step 3 visualizations
│   └── ...                     ← Other Python scripts
└── images/
    ├── input_image.jpg
    └── sift_step*.png          ← Visualization images
```

### Running the Code

```bash
# Main pipeline
python code/sift_pipeline.py

# Detailed Step 3 multi-octave visualization
python code/sift_step3_pyramid.py
```

---

## Detection Phase

**Goal**: Find stable, repeatable keypoints in the image that can be detected regardless of scale, rotation, or illumination changes.

```
INPUT: Image (H × W)
        ↓
Step 1: Build Gaussian Scale-Space Pyramid
        ↓
Step 2: Compute Difference of Gaussians (DoG)
        ↓
Step 3: Detect Keypoints (26-neighbor extrema) → 1124 keypoints
        ↓
Step 4: Filter & Refine Keypoints
        - Low Contrast Removal     → 1124 keypoints
        - Edge Response Removal    → 963 keypoints
        - Sub-pixel Refinement     → 847 keypoints
        ↓
OUTPUT: 847 stable keypoints with (x, y, scale)
```

---

## Step 1: Gaussian Scale-Space Pyramid

**Why do we need this?** To detect features at any scale, we must analyze the image at multiple resolutions.

### The Mathematics

The scale-space representation is created by convolving the image with Gaussian kernels of increasing width:

```
L(x,y,σ) = G(x,y,σ) * I(x,y)

G(x,y,σ) = (1/2πσ²) exp(-(x²+y²)/2σ²)
```

Where:
- `L(x,y,σ)` is the scale-space representation
- `G(x,y,σ)` is the Gaussian kernel
- `I(x,y)` is the input image
- `σ` is the scale parameter

![Step 1](images/sift_step1_gaussian_pyramid.png)

---

## Step 2: Difference of Gaussians (DoG)

**Why DoG?** The Difference of Gaussians approximates the Laplacian of Gaussian, which is an excellent blob detector.

### The Mathematics

```
D(x,y,σ) = L(x,y,kσ) - L(x,y,σ)
         ≈ (k-1)σ² ∇²G * I
```

By subtracting adjacent scales, we get a response that highlights blob-like structures at different sizes.

![Step 2](images/sift_step2_dog.png)

---

## Step 3: Keypoint Detection

Keypoint detection uses **26-neighbor comparison** across three consecutive DoG images.

### Step 3.1: Three DoG Scales

We need three consecutive DoG images for scale-space extrema detection:

![Step 3.1](images/sift_step3_1_three_scales.png)

### Step 3.2: Understanding the 26 Neighbors

For each pixel, we compare against **26 neighbors**:
- 9 at scale σ-1
- 8 at scale σ (same scale, exclude center)
- 9 at scale σ+1

```
Keypoint if:
  value > ALL 26 neighbors → Maximum
  value < ALL 26 neighbors → Minimum
```

![Step 3.2](images/sift_step3_2_26_neighbors.png)

### Step 3.3-3.5: Multi-Octave Processing

The process repeats at multiple octaves (resolutions):

| Octave | Resolution | Processing |
|--------|------------|------------|
| 0 | H × W | Full resolution keypoints |
| 1 | H/2 × W/2 | Half resolution, scaled back 2× |
| 2 | H/4 × W/4 | Quarter resolution, scaled back 4× |

![Step 3.3](images/sift_step3_3_octave0.png)
![Step 3.4](images/sift_step3_4_octave1.png)
![Step 3.5](images/sift_step3_5_octave2.png)

### Step 3.6: Complete Pyramid Structure

```
OCTAVE 0 (H×W):      G(σ₁) → G(σ₂) → G(σ₃) → G(σ₄)  →  DoG → 26-nbr → KP
    ↓ downsample
OCTAVE 1 (H/2×W/2):  G(σ₁) → G(σ₂) → G(σ₃) → G(σ₄)  →  DoG → 26-nbr → KP
    ↓ downsample
OCTAVE 2 (H/4×W/4):  G(σ₁) → G(σ₂) → G(σ₃) → G(σ₄)  →  DoG → 26-nbr → KP
```

![Step 3.6](images/sift_step3_6_pyramid_structure.png)

### Step 3.7: Combining Keypoints from All Octaves

Keypoints from different octaves must be scaled back to the original image coordinates:

| Octave | Resolution | Scale Factor | Coordinate Transform |
|--------|------------|--------------|---------------------|
| 0 | H × W | 1 | (x, y) → (x, y) |
| 1 | H/2 × W/2 | 2 | (x, y) → (x×2, y×2) |
| 2 | H/4 × W/4 | 4 | (x, y) → (x×4, y×4) |

**Implementation:**

```python
# Octave 0: no scaling
kp0 = detect_keypoints(gray)  # [(x, y, type), ...]

# Octave 1: scale by 2
gray1 = gray[::2, ::2]  # Downsample
kp1_local = detect_keypoints(gray1)
kp1 = [(x*2, y*2, t) for x, y, t in kp1_local]  # Scale back

# Octave 2: scale by 4
gray2 = gray[::4, ::4]  # Downsample
kp2_local = detect_keypoints(gray2)
kp2 = [(x*4, y*4, t) for x, y, t in kp2_local]  # Scale back

# Combine all
all_keypoints = kp0 + kp1 + kp2
```

![Scale Factor Visual](images/sift_scale_factor_real.png)

### All Octaves Combined

Circle size and color indicate detection scale:
- **Red small circles**: Octave 0 (Fine-scale) - 856 keypoints
- **Green medium circles**: Octave 1 (Medium-scale) - 213 keypoints
- **Blue large circles**: Octave 2 (Coarse-scale) - 55 keypoints

![All Octaves Combined](images/sift_all_octaves_combined.png)

```
OCTAVE 0 (H × W):      → 856 keypoints (Red, Small)
OCTAVE 1 (H/2 × W/2):  → 213 keypoints (Green, Medium)
OCTAVE 2 (H/4 × W/4):  →  55 keypoints (Blue, Large)
                       ─────────────
TOTAL DETECTED:        1124 keypoints
```

---

## Step 4: Keypoint Filtering & Refinement

The 1124 initial keypoints are filtered through three stages:

```
Step 3 Complete: 1124 keypoints detected
        ↓
Stage 1: Low Contrast Removal     |D(x̂)| < 0.03      → 1124 remaining
        ↓
Stage 2: Edge Response Removal    Tr(H)²/Det(H) > 12.1 → 963 remaining
        ↓
Stage 3: Sub-pixel Refinement     |offset| > 0.5      → 847 remaining
        ↓
FINAL: 847 stable keypoints (75.4% of 1124)
```

### Understanding the Filtering Coordinates

**(x, y) refers to keypoint coordinates, NOT all image pixels.**

```
Image size: 640 × 480 = 307,200 total pixels

Step 3 detected: 1124 keypoints (blob points)
  - Each keypoint has coordinates (x, y) where it was detected

Filtering is applied ONLY to these 1124 points:
  - For each keypoint:
    1. Look at 3×3 neighborhood around (x, y) in DoG image
    2. Compute derivatives using neighboring pixels
    3. Apply filtering tests
    4. Keep or reject
```

### Why a 3×3 Window?

We need a **3×3 window** around each keypoint to compute derivatives using **finite differences**:

```
Keypoint at (x, y):

       x-1    x    x+1
      ┌─────┬─────┬─────┐
y-1   │ NW  │  N  │ NE  │   ← Need y-1 for Dy, Dyy, Dxy
      ├─────┼─────┼─────┤
y     │  W  │  C  │  E  │   ← Need x-1, x+1 for Dx, Dxx
      ├─────┼─────┼─────┤
y+1   │ SW  │  S  │ SE  │   ← Need y+1 for Dy, Dyy, Dxy
      └─────┴─────┴─────┘

C = Center (keypoint location)
```

### Derivative Computations

**First Derivatives (Gradient):**

```
Dx = [D(x+1,y) - D(x-1,y)] / 2    ← uses E and W
Dy = [D(x,y+1) - D(x,y-1)] / 2    ← uses S and N
```

**Second Derivatives (Curvature):**

```
Dxx = D(x+1,y) + D(x-1,y) - 2×D(x,y)    ← uses E, W, C
Dyy = D(x,y+1) + D(x,y-1) - 2×D(x,y)    ← uses S, N, C
Dxy = [D(x+1,y+1) - D(x+1,y-1) - D(x-1,y+1) + D(x-1,y-1)] / 4
```

### Stage 1: Low Contrast Removal

**Purpose**: Remove keypoints sensitive to noise.

**Process:**

1. Compute sub-pixel offset using Hessian inverse
2. Calculate contrast at refined location
3. Reject if contrast is below threshold

```
D(x̂) = D(x,y) + 0.5 × (Dx × offset_x + Dy × offset_y)

REJECT if: |D(x̂)| < 0.03
```

**Example - Keypoint KEPT:**

```
DoG neighborhood around (50, 80):
       x=49   x=50   x=51
      ┌──────┬──────┬──────┐
y=79  │  15  │  18  │  12  │
      ├──────┼──────┼──────┤
y=80  │  20  │  45  │  25  │  ← Center = 45 (strong response)
      ├──────┼──────┼──────┤
y=81  │  14  │  22  │  16  │
      └──────┴──────┴──────┘

D(x̂) = 45.112
|45.112| < 0.03? NO → KEEP
```

![Stage 1 Low Contrast](images/sift_stage1_low_contrast.png)

### Stage 2: Edge Response Removal

**Purpose**: Remove keypoints on edges (poorly localized along edge direction).

**Mathematics:**

```
Hessian Matrix:
H = | Dxx  Dxy |
    | Dxy  Dyy |

Edge Ratio Test:
Tr(H)²/Det(H) > (r + 1)²/r

Default r = 10  →  Threshold = 12.1

REJECT if ratio > 12.1 (edge-like response)
KEEP if ratio ≤ 12.1 (blob-like response)
```

**Example - Edge Rejected:**

```
Keypoint at (350, 200):
Strong response along Y-axis but weak along X-axis → EDGE

Dxx = -59, Dyy = -3, Dxy = 0
Tr(H) = -62
Det(H) = 177
Ratio = (-62)² / 177 = 21.72 > 12.1 → REJECT
```

![Stage 2 Edge Response](images/sift_stage2_edge_response.png)

### Stage 3: Sub-pixel Refinement

**Purpose**: Remove unstable keypoints where the true extremum is in a different pixel.

```
REJECT if: |offset_x| > 0.5 OR |offset_y| > 0.5
```

**Example - Unstable Rejected:**

```
Keypoint at (80, 420):
offset_x = -0.86  ← Too large! True peak almost 1 pixel away.

RESULT: REJECTED - Unstable keypoint
```

![Stage 3 Subpixel](images/sift_stage3_subpixel.png)

### Detection Phase Complete

```
Summary:
  Started with:  1124 keypoints
  Stage 1:       -0 (low contrast)
  Stage 2:       -161 (edges)
  Stage 3:       -116 (unstable)
  Final:         847 keypoints (75.4% retention)
```

---

## Description Phase

**Goal**: Create unique, rotation-invariant, scale-invariant fingerprints (descriptors) for each detected keypoint.

```
INPUT: 847 stable keypoints with (x, y, scale)
        ↓
Step 5: Orientation Assignment
        - 36-bin gradient histogram around each keypoint
        ↓
Step 6: Descriptor Extraction
        - 16×16 region → 4×4 subregions → 8-bin histograms
        - 128-D vector
        ↓
OUTPUT: 847 keypoints with (x, y, scale, orientation, 128-D descriptor)
```

---

## Step 5: Orientation Assignment

**Purpose**: Achieve rotation invariance by assigning a dominant orientation to each keypoint.

### The Process

1. Take a region around the keypoint
2. Compute gradient magnitude and direction for each pixel
3. Build a 36-bin histogram (each bin = 10°)
4. Dominant peak → keypoint's orientation

### The Mathematics

```
For each pixel (x, y) in a region:

Gradient:
  Gx = I(x+1, y) - I(x-1, y)
  Gy = I(x, y+1) - I(x, y-1)

Magnitude:   m(x,y) = √(Gx² + Gy²)
Direction:   θ(x,y) = arctan(Gy / Gx)   → 0° to 360°

36-bin Histogram:
  - Each bin covers 10°
  - Weight = magnitude × Gaussian(distance from center)
  - Dominant direction = peak bin
```

![Orientation Assignment](images/sift_desc_orientation.png)

---

## Step 6: Descriptor Extraction

**Purpose**: Create a unique 128-dimensional fingerprint for matching across images.

### Descriptor Overview

![SIFT Descriptor Overview](images/sift_descriptor_overview.png)

### Step 6.1: Extract 16×16 Region

Center a 16×16 pixel region on the keypoint, rotated according to the dominant orientation:

```
16×16 Region:
  - Centered on keypoint position (x, y)
  - Rotated by dominant orientation θ
  - Total: 256 pixels
```

![16x16 Region](images/sift_desc_16x16.png)

### Step 6.2: Divide into 4×4 Subregions

```
┌────┬────┬────┬────┐
│ S0 │ S1 │ S2 │ S3 │   Each subregion = 4×4 pixels
├────┼────┼────┼────┤
│ S4 │ S5 │ S6 │ S7 │   Total subregions = 16
├────┼────┼────┼────┤
│ S8 │ S9 │S10 │S11 │   Each subregion → 8-bin histogram
├────┼────┼────┼────┤
│S12 │S13 │S14 │S15 │
└────┴────┴────┴────┘
```

![4x4 Grid](images/sift_desc_4x4grid.png)

### Step 6.3: Compute Gradient Directions

![Gradients](images/sift_desc_gradients.png)

### Step 6.4: Build 8-bin Histogram per Subregion

```
8 bins (45° each):
  Bin 0: 0° - 45°      Bin 4: 180° - 225°
  Bin 1: 45° - 90°     Bin 5: 225° - 270°
  Bin 2: 90° - 135°    Bin 6: 270° - 315°
  Bin 3: 135° - 180°   Bin 7: 315° - 360°

Result: 16 subregions × 8 bins = 128 values
```

![8-bin Histograms](images/sift_desc_histograms.png)

### Step 6.5: Create Final 128-D Descriptor

```
Descriptor Structure:
  [S0: b0-b7][S1: b0-b7]...[S15: b0-b7]
  ─────────────────────────────────────
                128 values

Normalization:
  1. L2 normalize: d = d / ||d||
  2. Clip values > 0.2 (reduce illumination effects)
  3. Re-normalize
```

![128-D Descriptor](images/sift_desc_final128.png)

### Complete Pipeline

![Descriptor Pipeline](images/sift_desc_pipeline_real.png)

---

## Summary

### Complete SIFT Pipeline

```
INPUT: Image (H × W)

═══════════════════════════════════════════════════════════════════
                        DETECTION PHASE
═══════════════════════════════════════════════════════════════════
        ↓
STEP 1: Gaussian Scale-Space Pyramid
        ↓
STEP 2: Difference of Gaussians (DoG)
        ↓
STEP 3: Keypoint Detection (26-neighbor extrema)
        ↓
STEP 4: Keypoint Filtering & Refinement → 847 keypoints

═══════════════════════════════════════════════════════════════════
                        DESCRIPTION PHASE
═══════════════════════════════════════════════════════════════════
        ↓
STEP 5: Orientation Assignment
        ↓
STEP 6: Descriptor Extraction → 128-D per keypoint
        ↓
OUTPUT: 847 keypoints with (x, y, scale, orientation, 128-D descriptor)
```

![Complete SIFT Pipeline](images/sift_complete_summary.png)

### Quick Reference: Filtering Formulas

| Stage | Formula | Threshold | Action |
|-------|---------|-----------|--------|
| Stage 1 | `D(x̂) = D + 0.5 × ∇D·offset` | \|D(x̂)\| < 0.03 | REJECT |
| Stage 2 | `Tr(H)²/Det(H)` | > 12.1 | REJECT |
| Stage 3 | `offset = -H⁻¹ × ∇D` | \|offset\| > 0.5 | REJECT |

### Key Properties

| Property | Value |
|----------|-------|
| Year | 2004 (Lowe) |
| Speed | Slower than SURF |
| Detection | DoG extrema → 1124 keypoints |
| Filtering | 1124 → 847 (75.4% retention) |
| Description | 128-D descriptor per keypoint |
| Key Innovation | Scale-space pyramid + sub-pixel accuracy |

---

## References

1. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints." International Journal of Computer Vision, 60(2), 91-110.
