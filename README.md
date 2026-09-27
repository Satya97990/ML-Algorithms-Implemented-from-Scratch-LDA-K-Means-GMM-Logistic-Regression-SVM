# ML Algorithms Implemented from Scratch — LDA, K-Means, GMM, Logistic Regression, SVM

This repository contains from-scratch (NumPy-only, no `sklearn` model classes) implementations of several classical machine learning algorithms, along with experiments comparing them against their library counterparts. Done as Assignment 1 for **DS216 — Machine Learning for Data Science**, IISc Bangalore.

## Contents

- `ML-Algorithms-Implemented-from-Scratch-LDA-K-Means-GMM-Logistic-Regression-SVM.ipynb` — full source notebook with code, plots, and outputs
- `ML-Algorithms-Implemented-from-Scratch-LDA-K-Means-GMM-Logistic-Regression-SVM.pdf` — rendered report with write-ups, tables, and figures

## Overview

### Q1: Linear Discriminant Analysis (LDA)
- Implemented LDA from scratch: manually computed within-class scatter (`S_W`) and between-class scatter (`S_B`) matrices, then solved the generalized eigenvalue problem for `S_W⁻¹ S_B` to find the optimal projection directions.
- **Homoscedastic case:** 3 synthetic classes (10D, 500 points each) sharing one covariance matrix, projected to 2D. Logistic Regression on the projection reached **95.67%** accuracy.
- **Heteroscedastic case:** each class given its own covariance matrix. Manual LDA accuracy dropped to **54.33%**, since a single linear projection can't accommodate differently shaped/rotated class covariances. A library **QDA** model on the full 10D data reached **99.00%**, since it fits a separate covariance per class and produces curved decision boundaries.

### Q2: Image Compression via K-Means and GMM
- Implemented **K-Means** (iterative assignment + centroid update) and a **Gaussian Mixture Model** trained with the **EM algorithm** (soft assignments, GMM initialized from K-Means centroids) for color quantization.
- Swept `K ∈ {2, 4, 6, 8, 10, 12}` per image and used the **elbow method** on reconstruction SSE to pick the optimal `K` for each of 5 sample images.
- Reconstructed images by replacing each pixel with its cluster's mean color; compared visual quality (K-Means → sharp/posterized edges, GMM → softer, more natural gradients).
- Studied **sensitivity to random initialization** across repeated runs, tracking iterations to convergence, PSNR, and SSIM. K-Means converged faster (<80 iterations) and was highly stable; GMM took longer (100+ iterations) with slightly more run-to-run variance, though SSIM stayed consistent across both.

### Q3: Breast Cancer Classification
- Implemented **Logistic Regression** (sigmoid + gradient descent on binary cross-entropy) and a **soft-margin SVM** solved via **Sequential Minimal Optimization (SMO)**, supporting linear, polynomial, and RBF kernels — all from scratch using only NumPy.
- Evaluated the effect of **PCA dimensionality reduction** (30D → 15D → 10D → 5D → 2D) on test accuracy for all four models (LR, SVM-Linear, SVM-Poly, SVM-RBF). Accuracy stayed above 90% even at heavy compression, indicating high feature redundancy in the dataset; best trade-off point was around 5D–10D.
- Visualized 2D decision boundaries for models trained directly on 2D PCA components vs. models trained on the full 30D space and sliced back to 2D via inverse PCA transform. RBF produced the most flexible, best-fitting boundary; support vectors were highlighted for each SVM variant.

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scikit-image
Pillow
```

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn scikit-image Pillow
```

## Usage

Open the notebook in Jupyter and run the cells in order:
```bash
jupyter notebook ML-Algorithms-Implemented-from-Scratch-LDA-K-Means-GMM-Logistic-Regression-SVM.ipynb
```

Note: the K-Means/GMM image compression section (Q2) expects a set of input images (`img1.JPEG` … `img5.JPEG`) available at the path referenced in the notebook — update the image directory path if running locally.

## Author

Satyajeet Kumar — M.Tech, Computational and Data Sciences, IISc Bangalore
