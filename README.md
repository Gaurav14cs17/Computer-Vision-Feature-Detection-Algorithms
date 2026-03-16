# Computer Vision Feature Detection Algorithms

*From-scratch implementations of classic feature detection and description algorithms*

---

This repository contains comprehensive implementations of four fundamental computer vision algorithms for feature detection and description. Each implementation includes detailed mathematical explanations, step-by-step visualizations, and working Python code.

## Algorithms Covered

| Algorithm | Year | Descriptor | Key Innovation | Speed |
|-----------|------|------------|----------------|-------|
| [SIFT](#sift) | 2004 | 128-D | Scale-space pyramid + DoG extrema | Baseline |
| [SURF](#surf) | 2006 | 64-D | Integral images + box filters | ~3× faster |
| [ORB](#orb) | 2011 | 256-bit | FAST corners + rotated BRIEF | ~100× faster |
| [HOG](#hog) | 2005 | 167,796-D | Dense gradient histograms | N/A (dense) |

---

## Project Structure

```
AlgoImplementation/
├── README.md                    ← This file
├── 01_sift/
│   ├── README.md                ← Detailed SIFT documentation
│   ├── SIFT_Algorithm.ipynb     ← Interactive notebook
│   ├── code/                    ← Python implementations
│   │   ├── sift_pipeline.py     ← Main pipeline
│   │   └── ...                  ← Step-by-step scripts
│   └── images/                  ← Generated visualizations
├── 02_surf/
│   ├── README.md                ← Detailed SURF documentation
│   ├── SURF_Algorithm.ipynb     ← Interactive notebook
│   ├── code/                    ← Python implementations
│   │   ├── surf_pipeline.py     ← Main pipeline
│   │   └── ...                  ← Step-by-step scripts
│   └── images/                  ← Generated visualizations
├── 03_orb/
│   ├── README.md                ← Detailed ORB documentation
│   ├── ORB_Algorithm.ipynb      ← Interactive notebook
│   ├── code/                    ← Python implementations
│   │   ├── orb_pipeline.py      ← Main pipeline
│   │   └── ...                  ← Step-by-step scripts
│   └── images/                  ← Generated visualizations
└── 04_hog/
    ├── README.md                ← Detailed HOG documentation
    ├── HOG_Algorithm.ipynb      ← Interactive notebook
    ├── code/                    ← Python implementations
    │   ├── hog_pipeline.py      ← Main pipeline
    │   └── ...                  ← Step-by-step scripts
    └── images/                  ← Generated visualizations
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib pillow
```

### Running the Pipelines

```bash
# SIFT - Scale-Invariant Feature Transform
python 01_sift/code/sift_pipeline.py

# SURF - Speeded-Up Robust Features
python 02_surf/code/surf_pipeline.py

# ORB - Oriented FAST and Rotated BRIEF
python 03_orb/code/orb_pipeline.py

# HOG - Histogram of Oriented Gradients
python 04_hog/code/hog_pipeline.py
```

---

## Algorithm Summaries

### SIFT

**Scale-Invariant Feature Transform** (Lowe, 2004)

The foundational algorithm for scale and rotation invariant feature detection.

| Phase | Steps |
|-------|-------|
| Detection | Gaussian pyramid → DoG → 26-neighbor extrema → Keypoint filtering |
| Description | Orientation assignment → 128-D descriptor (4×4×8 bins) |

**Key Formulas:**
- Scale-space: `L(x,y,σ) = G(x,y,σ) * I(x,y)`
- DoG: `D(x,y,σ) = L(x,y,kσ) - L(x,y,σ)`
- Keypoint test: Compare against 26 neighbors across 3 scales

[→ Full SIFT Documentation](01_sift/README.md)

---

### SURF

**Speeded-Up Robust Features** (Bay et al., 2006)

A faster alternative to SIFT using integral images and box filters.

| Phase | Steps |
|-------|-------|
| Detection | Integral image → Hessian determinant → 26-neighbor extrema |
| Description | Haar wavelets orientation → 64-D descriptor (4×4×4 values) |

**Key Formulas:**
- Integral image: `II(x,y) = Σ I(i,j)` for all i≤x, j≤y
- Box sum (O(1)): `Sum = II(D) - II(B) - II(C) + II(A)`
- Hessian: `det(H) = Dxx·Dyy - (0.9·Dxy)²`

[→ Full SURF Documentation](02_surf/README.md)

---

### ORB

**Oriented FAST and Rotated BRIEF** (Rublee et al., 2011)

A real-time, patent-free alternative combining FAST detection with binary descriptors.

| Phase | Steps |
|-------|-------|
| Detection | Scale pyramid → FAST corners → Harris response → Orientation |
| Description | Rotated BRIEF (rBRIEF) → 256-bit binary descriptor |

**Key Formulas:**
- FAST test: 9+ contiguous pixels brighter/darker than center ± threshold
- Harris response: `R = det(M) - k·trace(M)²`
- Orientation: `θ = atan2(m₀₁, m₁₀)` using intensity centroid
- Matching: Hamming distance (XOR + popcount)

[→ Full ORB Documentation](03_orb/README.md)

---

### HOG

**Histogram of Oriented Gradients** (Dalal & Triggs, 2005)

A dense feature descriptor primarily used for object detection (especially pedestrians).

| Step | Description | Output |
|------|-------------|--------|
| 1 | Preprocessing | Grayscale + gamma correction |
| 2 | Gradients | Magnitude and direction per pixel |
| 3 | Cell histograms | 8×8 pixel cells → 9-bin histograms |
| 4 | Block normalization | 2×2 cells → L2 normalized |

**Key Formulas:**
- Gradient: `Gx = I(x+1,y) - I(x-1,y)`, `Gy = I(x,y+1) - I(x,y-1)`
- Magnitude: `m = √(Gx² + Gy²)`
- Direction: `θ = atan2(Gy, Gx)`

[→ Full HOG Documentation](04_hog/README.md)

---

## Algorithm Comparison

### Detection Methods

| Algorithm | Detector | Scale Invariance | Rotation Invariance |
|-----------|----------|------------------|---------------------|
| SIFT | DoG extrema | Yes (pyramid) | Yes (orientation) |
| SURF | Hessian determinant | Yes (filter pyramid) | Yes (Haar wavelets) |
| ORB | FAST corners | Yes (image pyramid) | Yes (intensity centroid) |
| HOG | Dense (no keypoints) | No (fixed window) | No (fixed bins) |

### Descriptor Properties

| Algorithm | Dimensions | Type | Matching Method |
|-----------|------------|------|-----------------|
| SIFT | 128 | Float | L2 distance |
| SURF | 64 | Float | L2 distance |
| ORB | 256 | Binary | Hamming distance |
| HOG | 167,796 | Float | SVM / L2 distance |

### Performance Trade-offs

| Algorithm | Speed | Accuracy | Patent-Free |
|-----------|-------|----------|-------------|
| SIFT | Slow | Excellent | No (expired 2020) |
| SURF | Fast | Very Good | No |
| ORB | Very Fast | Good | Yes |
| HOG | Moderate | Excellent (for detection) | Yes |

---

## Use Cases

| Application | Recommended Algorithm |
|-------------|----------------------|
| Image matching (high accuracy) | SIFT |
| Real-time feature tracking | ORB |
| 3D reconstruction | SIFT or SURF |
| Object detection | HOG |
| Mobile/embedded applications | ORB |
| Panorama stitching | SIFT or SURF |

---

## Learning Path

For those new to feature detection, we recommend this learning order:

1. **SIFT** - Start here to understand the foundational concepts of scale-space, keypoint detection, and descriptor extraction
2. **SURF** - Learn how integral images and box filters provide speedups while maintaining accuracy
3. **ORB** - Understand binary descriptors and real-time optimizations
4. **HOG** - Explore dense descriptors for object detection tasks

Each algorithm directory contains:
- A comprehensive README with mathematical explanations
- A Jupyter notebook for interactive exploration
- Multiple Python scripts showing step-by-step implementations
- Generated visualization images

---

## References

1. **SIFT**: Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints." IJCV, 60(2), 91-110.

2. **SURF**: Bay, H., Tuytelaars, T., & Van Gool, L. (2006). "SURF: Speeded Up Robust Features." ECCV 2006.

3. **ORB**: Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). "ORB: An efficient alternative to SIFT or SURF." ICCV 2011.

4. **HOG**: Dalal, N., & Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection." CVPR 2005.

---

## License

This project is for educational purposes. The algorithms are implementations based on the original papers cited above.
