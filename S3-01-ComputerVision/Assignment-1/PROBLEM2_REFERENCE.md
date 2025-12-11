# Problem 2: Lane Detection Using Classical Computer Vision + ML

## Overview
**Goal:** Develop a robust system for detecting straight lane lines in road images using purely classical computer vision and Machine Learning techniques.

**Core Techniques Required:**
- Edge Detection
- Hough Transformation
- Data Clustering/Averaging for line fitting

## Dataset
**Source:** [Lane Detection Dataset on Kaggle](https://www.kaggle.com/datasets/dataclusterlabs/lane-detection-road-line-detection-image-dataset?resource=download)

**TODO:** Download and organize the dataset

---

## Assignment Tasks Breakdown

### Part 1: Preprocessing and Edge Detection

#### A. Load and Convert
- [ ] Load sample image
- [ ] Convert to grayscale
- [ ] Apply Gaussian Blur to reduce noise
- [ ] **Experiment:** Convert to HLS or HSV color space to enhance white/yellow line visibility

#### B. Canny Edge Detection
- [ ] Apply Canny Edge Detector
- [ ] Document chosen thresholds: τ_high and τ_low
- [ ] Experiment with threshold values for best results

#### C. Region of Interest (ROI)
- [ ] Define trapezoidal or triangular polygonal mask
- [ ] Apply mask to Canny edge map
- [ ] Focus computation only on road area in front of vehicle

**Key Deliverable:** Preprocessed images with edge detection applied

---

### Part 2: Line Detection via Hough Transformation

#### A. Hough Transform
- [ ] Apply Hough Transform to masked edge image
- [ ] Get line segments defined by parameters (ρ, θ) or (x1, y1, x2, y2)
- [ ] Document Hough parameters used

#### B. Parameter Space Explanation
**Mathematical equation:** ρ = x cos θ + y sin θ

- [ ] Explain how a line in image space (x, y) maps to a point in Hough space (ρ, θ)
- [ ] Include visual demonstration if possible

**Key Deliverable:** Detected line segments from Hough Transform

---

### Part 3: Data Clustering and Lane Averaging (ML Focus)

This is the **CORE ML COMPONENT** of the assignment.

#### A. Slope Calculation and Classification
For each detected line segment:
- [ ] Calculate slope (m)
- [ ] Classify into groups:
  - **Left Lane Group:** Lines with negative slope (m < -ε)
  - **Right Lane Group:** Lines with positive slope (m > +ε)
- [ ] Discard near-zero slopes (|m| ≤ ε) as noise/irrelevant horizontal features

#### B. ML-Based Line Fitting
Choose ONE of the following methods for each group (Left and Right):

**Option 1: Simple Averaging**
- [ ] Calculate mean slope (m̄) for all lines in group
- [ ] Calculate mean intercept (b̄) for all lines in group
- [ ] Result: Single representative line per group

**Option 2: Robust Fitting (Advanced ML)**
- [ ] Implement RANSAC (Random Sample Consensus) algorithm
  - OR
- [ ] Apply k-means clustering (k=1 for each group) on (slope, intercept) feature space
- [ ] Find best-fit line parameters (m̄, b̄) robust to outliers

#### C. Projection
- [ ] Use final calculated line parameters:
  - Left: (m̄_left, b̄_left)
  - Right: (m̄_right, b̄_right)
- [ ] Extrapolate and draw two lane lines onto original color image
- [ ] Overlay lines over the defined ROI

**Key Deliverable:** Final images with detected lane lines overlaid

---

## Deliverables Checklist

### 1. Source Code
- [ ] Fully commented Python code
- [ ] Implement entire pipeline
- [ ] Use appropriate libraries (OpenCV, NumPy, scikit-learn)
- [ ] Modular and readable code structure

### 2. Output Images
- [ ] Process at least 3 different test cases
- [ ] Show detected and overlaid lane lines clearly
- [ ] Include intermediate visualizations (edges, ROI, Hough lines)

### 3. Technical Report (PDF)
- [ ] Document Canny threshold values chosen
- [ ] Document Hough Transform parameters
- [ ] Provide mathematical justification for chosen ML-based line fitting method (Part 3b)
- [ ] Analyze system performance
- [ ] Discuss observed limitations:
  - Curved roads
  - Shadows
  - Different lighting conditions
  - Occlusions

---

## Grading Rubric (Total: 10 points)

| Criteria | Description | Points |
|----------|-------------|--------|
| Import Required Libraries | OpenCV, skimage, scikit-learn, pandas, matplotlib with comments | 0.5 |
| Data Acquisition | Load dataset, print size, category-wise count, plot distribution | 0.5 |
| Data Preparation | Resize, grayscale, histogram equalization, 80-20 stratified split | 1.0 |
| Feature Engineering | LBP, HOG, Edge detection, Normalization, Optional: PCA | 2.5 |
| Model Building | Classical ML (SVM/Random Forest/XGBoost), training, cross-validation | 1.5 |
| Validation Metrics | Accuracy and F1-score for test data | 0.5 |
| Model Inference & Evaluation | 5 random test images, predicted vs actual, justification | 1.0 |
| Validation of actual test | Test image created by you, justify performance | 1.5 |
| Documentation & Code Quality | Well-documented, clear presentation, clean code | 1.0 |

---

## Implementation Tips

### Recommended Libraries
```python
import cv2  # OpenCV for image processing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
```

### Typical Pipeline Structure
1. **Image Loading** → Color conversion → Gaussian Blur
2. **Edge Detection** → Canny with tuned thresholds
3. **ROI Masking** → Apply polygon mask
4. **Hough Transform** → Detect line segments
5. **Slope Filtering** → Separate left/right lanes
6. **Line Fitting** → Average or RANSAC/K-means
7. **Visualization** → Draw final lanes on original image

### Common Pitfalls to Avoid
- Not tuning Canny thresholds properly
- ROI mask too restrictive or too loose
- Not filtering out horizontal lines (near-zero slopes)
- Not handling edge cases (no lines detected)
- Poor visualization of results

---

## Progress Tracker

### Phase 1: Setup and Understanding
- [ ] Dataset downloaded and organized
- [ ] Environment setup (libraries installed)
- [ ] Understood assignment requirements
- [ ] Read through reference papers/tutorials on Hough Transform

### Phase 2: Implementation
- [ ] Part 1 completed (Preprocessing and Edge Detection)
- [ ] Part 2 completed (Hough Transformation)
- [ ] Part 3 completed (ML-based Line Fitting)

### Phase 3: Testing and Refinement
- [ ] Tested on 3+ different images
- [ ] Parameter tuning completed
- [ ] Edge cases handled

### Phase 4: Documentation
- [ ] Code fully commented
- [ ] Technical report written
- [ ] Results visualized
- [ ] Individual contributions listed

### Phase 5: Submission
- [ ] Jupyter notebook with outputs
- [ ] Converted to PDF/HTML
- [ ] File named correctly: CV_assignment1_group_2
- [ ] Submitted on time

---

## Key Concepts to Understand

### Hough Transform
The Hough Transform detects lines by transforming points in image space to curves in parameter space. Each point (x, y) in image space votes for all possible lines passing through it using:

**ρ = x cos θ + y sin θ**

Where:
- ρ = perpendicular distance from origin to the line
- θ = angle of the perpendicular with x-axis

Lines appear as peaks in the Hough accumulator space.

### RANSAC (Random Sample Consensus)
Iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers.

**Algorithm:**
1. Randomly select minimum number of points
2. Fit model to selected points
3. Count inliers (points within threshold)
4. Repeat N iterations
5. Keep model with most inliers

### K-Means Clustering
Partitions data into k clusters by minimizing within-cluster variance.

For lane detection: Use k=1 for each lane group to find the centroid (mean slope, mean intercept).

---

## Notes and Observations

*(Use this section to track your experiments, observations, and decisions while working)*

**Date:** [Add date when you start]

**Session 1:**
- [Track what you worked on]
- [Parameters tried]
- [Results observed]
- [Next steps]

**Session 2:**
- ...

---

## Resources

### Helpful Tutorials
- OpenCV Hough Transform: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
- Canny Edge Detection: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

### Key Functions to Use
- `cv2.Canny()` - Edge detection
- `cv2.HoughLinesP()` - Probabilistic Hough Transform
- `cv2.line()` - Draw lines
- `cv2.fillPoly()` - Create ROI mask
- `cv2.addWeighted()` - Overlay lines on original image

---

## Questions to Answer in Report

1. Why did you choose specific Canny threshold values?
2. How did you determine the ROI shape and position?
3. What Hough Transform parameters worked best? Why?
4. Which ML-based line fitting method did you use? Justify mathematically.
5. What are the main limitations of your approach?
6. How would you handle curved roads?
7. How do different lighting conditions affect performance?

---

## Submission Checklist

Before submitting, verify:
- [ ] No plagiarism (original work)
- [ ] Individual contributions listed at end
- [ ] File naming convention followed
- [ ] Outputs are complete and properly aligned
- [ ] Code is well-commented
- [ ] No excessively long prints
- [ ] Notebook converted to PDF/HTML with outputs
- [ ] Latest submission (if multiple submissions made)
