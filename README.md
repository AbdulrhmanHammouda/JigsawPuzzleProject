

# 🧩 Jigsaw Puzzle Image Processing & Edge Detection – Phase 1

This repository contains **Phase 1** of a Jigsaw Puzzle Solver project.
The goal of this phase is to **prepare** the dataset of puzzle images by applying:

✔ Image enhancement
✔ Image slicing
✔ Edge detection
✔ Visualization & dataset structuring

These processed images will later power **puzzle assembly (Phase 2)**.

---

## 📁 Project Overview

This project processes a dataset of cartoon/anime puzzle images (`2×2`, `4×4`, `8×8`) and generates:

### **1. Enhanced Images**

Light smoothing → optional sharpening → slight saturation boost
(you used a simple pipeline to avoid losing important puzzle edge information)

### **2. Sliced Images**

Each enhanced image is divided into tiles (2×2, 4×4, 8×8) and saved individually.

### **3. Edge Detection Outputs**

Applied **three classic edge detectors** on every sliced tile:

* **Sobel**
* **Laplacian**
* **Canny**

These are stored in:

```
phase1/results/edges_detection_results/
    ├── sobel/
    ├── laplacian/
    └── canny/
```

---

## 🏗 Folder Structure (important parts)

```
phase1/
│
├── utils/
│   ├── enhancement.py           # enhancement pipeline
│   ├── slicing.py               # slicing into grid tiles
│   ├── edges.py                 # sobel / laplacian / canny
│   └── visualization.py
│
├── results/
│   ├── enhanced_images/         # enhanced full images
│   ├── enhanced_images_sliced/  # enhanced → sliced tiles
│   └── edges_detection_results/ # sobel, laplacian, canny outputs
│
├── report/
│
└── Phase1.ipynb                 # main notebook (pipeline runner)
```

---

## 🔧 Main Features

### **✓ Image Enhancement**

A light and safe enhancement pipeline:

```python
smooth → sharpen → slight saturation boost
```

Designed to improve **edge clarity** without destroying puzzle details.

### **✓ Automatic Slicing**

Each enhanced image is sliced according to its grid size:

* `puzzle_2x2` → 4 pieces
* `puzzle_4x4` → 16 pieces
* `puzzle_8x8` → 64 pieces

Tiles are saved for further edge matching.

### **✓ Edge Detection**

Three edge detectors:

```
Sobel (gradient magnitude)
Laplacian (second derivative)
Canny (binary edges)
```

Outputs are saved per puzzle folder.

---

## ▶ Running the Project

### **1. Enhancement + Slicing**

Run the main notebook:

```
Phase1.ipynb
```

This generates:

* `enhanced_images`
* `enhanced_images_sliced`

### **2. Edge Detection**

Run:

```bash
python edges.py
```

This generates:

```
results/edges_detection_results/
```

With subfolders for:

* Sobel
* Laplacian
* Canny

---

## 🖼 Sample Outputs

### **Edge Detection Example**

| Sobel                    | Laplacian               | Canny              |
| ------------------------ | ----------------------- | ------------------ |
| Grayscale gradient edges | Second derivative edges | Clean binary edges |

---

## 🧠 Why This Processing Matters (for Phase 2)

Puzzle assembly later depends heavily on:

* Edge clarity
* Texture consistency
* Accurate borders
* Reduced noise

Phase 1 ensures each tile is **clean, consistent, and edge-highlighted**, making reconstruction algorithms much more accurate.

---

## 👨‍💻 Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

---

## 📌 Future Work (Phase 2)

* Piece matching (border compatibility)
* Puzzle assembly algorithms (2×2, 4×4, 8×8)
* Correctness evaluation
* Optimization for speed & accuracy

---

## 📬 Author

Made with 😵, 😤, ☕, and lots of debugging by **Abdulrhman Hammouda**.
