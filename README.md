 

# 🧩 Jigsaw Puzzle Solver – Image Processing & Assembly (Phase 1 & 2)

This project implements a **classical computer vision pipeline** to solve square-piece jigsaw puzzles **without using AI/ML models**. It is divided into two phases:

* **Phase 1:** Dataset preparation and image processing
* **Phase 2:** Puzzle assembly using edge matching and heuristic search

---

## 📁 Project Overview

This system works with puzzle grids of sizes `2×2`, `4×4`, and `8×8`, and performs:

### 🔹 Phase 1 – Image Preprocessing

1. **Image Enhancement:** smoothing + sharpening + saturation boost
2. **Image Slicing:** cuts each puzzle into equal square tiles
3. **Edge Detection:** classic operators (`Sobel`, `Laplacian`, `Canny`) on each tile

All results are saved in a structured folder for reuse.

### 🔹 Phase 2 – Puzzle Assembly

Given sliced and enhanced tiles, the system reconstructs the puzzle using:

* Edge representation (LAB color, gradients)
* Brute-force search (2×2) or heuristic solvers (4×4+, BB matching, beam search)

---

## 📊 Results Summary

| Puzzle Size         | Accuracy             | Notebook           |
| ------------------- | -------------------- | ------------------ |
| **2×2** (4 pieces)  | **99.09%** (109/110) | `2x2_solver.ipynb` |
| **4×4** (16 pieces) | **92.73%** (102/110) | `4x4_solver.ipynb` |
| **8×8** (64 pieces) | **5.45%** (6/110)    | `8x8_solver.ipynb` |

---

## 🧩 Phase 1: Preprocessing & Edge Detection

### 📦 Outputs

Each original image produces:

* Enhanced version
* Sliced tiles
* Edge versions of tiles (Sobel, Laplacian, Canny)

### 📁 Folder Structure

```
phase1/results/
├── enhanced_images_sliced/
│   ├── puzzle_2x2/
│   ├── puzzle_4x4/
│   └── puzzle_8x8/
└── edges_detection_results/
    ├── sobel/
    ├── laplacian/
    └── canny/
```

---

## 🧩 Phase 2: Solvers

### ✅ 2×2 Solver (`2x2_solver.ipynb`)

* **Algorithm:** Brute-force (4! = 24 permutations)
* **Edge Features:** LAB + Sobel Gradient + Laplacian
* **Matching Metric:** Weighted distance (0.5 LAB, 0.3 Gradient, 0.2 Laplacian)
* **Output Folder:** `results/2x2_out/`

#### 🔧 Functions:

```python
edge_features(edge)     # Extract color & edge features
edge_distance(a, b)     # Computes distance
solve_2x2(pieces)       # Try all permutations
```

---

### 🔲 4×4 Solver (`4x4_solver.ipynb`)

* **Algorithm:** Heuristic Placer → Best-Buddies Refinement → Shifter (greedy reseeding)
* **Features:** LAB + Gradients + Laplacian
* **Matching:** Best-Buddies priority + compatibility matrix
* **Refinement:** Region growing + swaps
* **Output Folder:** `results/4x4_out/`

#### 🔧 Pipeline:

```python
extract_borders(piece)        # LAB + gradient strips
border_distance_2d(a, b)      # Edge comparison
build_compatibility(pieces)   # Cost matrix (4 directions)
placer(n, grid_n, compat)     # Greedy BB placement
shifter(..., compat)          # Segment grow + reseed
```

---

### ⬛ 8×8 Solver (`8x8_solver.ipynb`)

* **Algorithm:** Meta-solver ensemble (5 strategies) + SA refinement

* **Strategies Include:**

  * LAB Beam
  * Best-Buddy Guided Greedy
  * Hybrid LAB + Gradient
  * Wide Beam (beam=300)

* **Output Folder:** `results/8x8_out/`

---

## 🖼️ Visualization & Demonstration

* Each notebook includes side-by-side visual outputs:

  * Original image
  * Final reconstructed layout
* Intermediate debugging outputs include:

  * Edge strips
  * Matching compatibility maps
  * Segmentation clusters (4×4+)

> For submission: include one clean puzzle and one challenging case (rotation, noise).

---

## 🧠 Key Concepts

### Best Buddies

Two tiles A and B are best buddies if:

* A’s right edge best matches B’s left, **and**
* B’s left edge best matches A’s right

### LAB Color Space

Used instead of RGB for more perceptually uniform distance calculations.

### Beam Search

Keeps only top-K partial solutions at each step to manage combinatorics.

### Simulated Annealing (SA)

Probabilistically accepts worse solutions to escape local minima.


 

