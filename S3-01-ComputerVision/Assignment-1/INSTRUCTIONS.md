# IMMEDIATE FIX - No Kernel Restart Needed

The issue is that your Jupyter kernel has the **old function cached in memory**.

Even though we updated the notebook file, the kernel is still using the old version.

## Solution: Run This Code Directly in Jupyter

### Step 1: Open Your Notebook

Open **either**:
- `CV_Assignment1_Group_Problem2.ipynb` (main assignment), OR
- `MySolution.ipynb` (your working copy)

### Step 2: Create a New Cell

**Insert a new cell AFTER the current `detect_lanes` definition and BEFORE any test cells**

### Step 3: Copy-Paste This Entire Function

```python
import random
import numpy as np
import cv2

def detect_lanes(image, config=None, debug=False):
    """
    Complete end-to-end lane detection pipeline - FIXED VERSION
    """

    # FIXED CONFIGURATION - Relaxed parameters
    if config is None:
        config = {
            'gaussian_kernel': (5, 5),
            'gaussian_sigma': 1.5,
            'canny_low': 30,              # FIXED: Was 50, now 30
            'canny_high': 100,            # FIXED: Was 150, now 100
            'roi_bottom_width_ratio': 0.90,
            'roi_top_width_ratio': 0.20,  # FIXED: Was 0.08, now 0.20 (CRITICAL)
            'roi_top_y_ratio': 0.55,      # FIXED: Was 0.60, now 0.55
            'roi_bottom_y_ratio': 0.98,
            'hough_rho': 1,               # FIXED: Was 2, now 1
            'hough_theta': np.pi / 180,
            'hough_threshold': 10,        # FIXED: Was 50, now 10 (CRITICAL)
            'hough_min_line_length': 20,  # FIXED: Was 50, now 20 (CRITICAL)
            'hough_max_line_gap': 200,    # FIXED: Was 50, now 200 (CRITICAL)
            'slope_threshold': 0.4,       # FIXED: Was 0.5, now 0.4
            'ransac_iterations': 1000,
            'ransac_distance_threshold': 25,
            'ransac_min_inliers_ratio': 0.4,
        }

    debug_images = {}
    height, width = image.shape[:2]

    # Stage 1: Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, config['gaussian_kernel'], config['gaussian_sigma'])
    if debug:
        debug_images['1_preprocessed'] = blurred

    # Stage 2: Canny Edge Detection
    edges = cv2.Canny(blurred, config['canny_low'], config['canny_high'])
    if debug:
        debug_images['2_edges'] = edges

    # Stage 3: ROI Masking
    bottom_left = (int(width * (1 - config['roi_bottom_width_ratio']) / 2),
                   int(height * config['roi_bottom_y_ratio']))
    bottom_right = (int(width * (1 + config['roi_bottom_width_ratio']) / 2),
                    int(height * config['roi_bottom_y_ratio']))
    top_left = (int(width * (1 - config['roi_top_width_ratio']) / 2),
                int(height * config['roi_top_y_ratio']))
    top_right = (int(width * (1 + config['roi_top_width_ratio']) / 2),
                 int(height * config['roi_top_y_ratio']))

    roi_vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    if debug:
        debug_images['3_masked_edges'] = masked_edges
        print(f"[DEBUG] Edge pixels in ROI: {np.count_nonzero(masked_edges)}")

    # Stage 4: Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        config['hough_rho'],
        config['hough_theta'],
        config['hough_threshold'],
        minLineLength=config['hough_min_line_length'],
        maxLineGap=config['hough_max_line_gap']
    )

    if debug:
        print(f"[DEBUG] Hough detected {len(lines) if lines is not None else 0} lines")

    if lines is None or len(lines) == 0:
        if debug:
            print("⚠️  No lines detected by Hough Transform")
        return {
            'annotated_image': image.copy(),
            'left_line': None,
            'right_line': None,
            'left_confidence': 0,
            'right_confidence': 0,
            'debug_images': debug_images if debug else None
        }

    # Stage 5: Lane Separation
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)

        if slope < -config['slope_threshold']:
            left_lines.append(line[0])
        elif slope > config['slope_threshold']:
            right_lines.append(line[0])

    if debug:
        print(f"[DEBUG] Left candidates: {len(left_lines)}, Right candidates: {len(right_lines)}")

    # Stage 6: RANSAC Fitting
    def lines_to_points(lines):
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 100:
                points.append(((x1 + x2) // 2, (y1 + y2) // 2))
        return points

    def ransac_line_fit(points, max_iterations, distance_threshold, min_inliers_ratio):
        if len(points) < 2:
            return None, None, [], []

        min_inliers = int(len(points) * min_inliers_ratio)
        best_slope, best_intercept = None, None
        best_inliers = []
        best_inlier_count = 0

        for _ in range(max_iterations):
            sample = random.sample(points, 2)
            (x1, y1), (x2, y2) = sample
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            inliers = [p for p in points if abs(p[1] - (slope * p[0] + intercept)) < distance_threshold]

            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_slope = slope
                best_intercept = intercept
                best_inliers = inliers

        if best_inlier_count < min_inliers:
            return None, None, [], []

        if len(best_inliers) >= 2:
            x_coords = np.array([p[0] for p in best_inliers])
            y_coords = np.array([p[1] for p in best_inliers])
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            best_slope, best_intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]

        return best_slope, best_intercept, best_inliers, []

    left_slope, left_intercept, left_confidence = None, None, 0
    right_slope, right_intercept, right_confidence = None, None, 0

    if len(left_lines) > 0:
        left_points = lines_to_points(left_lines)
        left_slope, left_intercept, left_inliers, _ = ransac_line_fit(
            left_points, config['ransac_iterations'],
            config['ransac_distance_threshold'], config['ransac_min_inliers_ratio']
        )
        if left_slope is not None:
            left_confidence = len(left_inliers) / len(left_points)
            if debug:
                print(f"[DEBUG] Left: y={left_slope:.4f}x+{left_intercept:.2f}, conf={left_confidence:.1%}")

    if len(right_lines) > 0:
        right_points = lines_to_points(right_lines)
        right_slope, right_intercept, right_inliers, _ = ransac_line_fit(
            right_points, config['ransac_iterations'],
            config['ransac_distance_threshold'], config['ransac_min_inliers_ratio']
        )
        if right_slope is not None:
            right_confidence = len(right_inliers) / len(right_points)
            if debug:
                print(f"[DEBUG] Right: y={right_slope:.4f}x+{right_intercept:.2f}, conf={right_confidence:.1%}")

    # Stage 7: Draw lanes
    y_top = int(height * config['roi_top_y_ratio'])
    y_bottom = int(height * config['roi_bottom_y_ratio'])

    annotated = image.copy()
    overlay = annotated.copy()

    if left_slope is not None and left_intercept is not None:
        try:
            x1 = int((y_top - left_intercept) / left_slope)
            x2 = int((y_bottom - left_intercept) / left_slope)
            cv2.line(overlay, (x1, y_top), (x2, y_bottom), (255, 0, 0), 12)
        except:
            pass

    if right_slope is not None and right_intercept is not None:
        try:
            x1 = int((y_top - right_intercept) / right_slope)
            x2 = int((y_bottom - right_intercept) / right_slope)
            cv2.line(overlay, (x1, y_top), (x2, y_bottom), (0, 255, 255), 12)
        except:
            pass

    annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
    if debug:
        debug_images['4_final'] = annotated

    return {
        'annotated_image': annotated,
        'left_line': (left_slope, left_intercept) if left_slope is not None else None,
        'right_line': (right_slope, right_intercept) if right_slope is not None else None,
        'left_confidence': left_confidence,
        'right_confidence': right_confidence,
        'debug_images': debug_images if debug else None
    }

print("✅ FIXED detect_lanes loaded! Re-run your tests now.")
print("Key changes: threshold 50→10, minLen 50→20, maxGap 50→200, ROI top 0.08→0.20")
```

### Step 4: Run the Cell

Press **Shift+Enter** to run it. You should see:
```
✅ FIXED detect_lanes loaded! Re-run your tests now.
```

### Step 5: Quick Test

Add another new cell and run this:

```python
# Quick test
test_result = detect_lanes(sample_images[0], debug=True)
print(f"\n{'='*50}")
print(f"Left:  {'✅ DETECTED' if test_result['left_line'] else '❌ FAILED'}")
print(f"Right: {'✅ DETECTED' if test_result['right_line'] else '❌ FAILED'}")
print(f"{'='*50}")
```

You should now see **at least one lane detected** (hopefully both!).

### Step 6: Re-run All Tests

Now re-run your Section 9 and Section 10 test cells. Detection rate should jump from 12.5% to **60-90%**.

---

## What Changed

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `hough_threshold` | 50 | **10** | 80% reduction - detects weaker lines |
| `hough_min_line_length` | 50 | **20** | 60% reduction - accepts shorter segments |
| `hough_max_line_gap` | 50 | **200** | 300% increase - connects dashed lanes |
| `roi_top_width_ratio` | 0.08 | **0.20** | 150% wider - captures lanes at horizon |
| `canny_low` | 50 | **30** | More sensitive edge detection |
| `canny_high` | 150 | **100** | More sensitive edge detection |
| `ransac_min_inliers_ratio` | 0.5 | **0.4** | More tolerant to outliers |

---

## If Still Failing

If you're still getting 0% detection after running the fixed function:

1. **Check image loading:**
   ```python
   print(f"sample_images length: {len(sample_images)}")
   print(f"First image shape: {sample_images[0].shape}")
   plt.imshow(cv2.cvtColor(sample_images[0], cv2.COLOR_BGR2RGB))
   plt.title("First Test Image")
   plt.show()
   ```

2. **Try ultra-extreme parameters:**
   ```python
   extreme_config = {
       'gaussian_kernel': (5, 5),
       'gaussian_sigma': 1.5,
       'canny_low': 20,
       'canny_high': 80,
       'roi_bottom_width_ratio': 0.95,
       'roi_top_width_ratio': 0.30,
       'roi_top_y_ratio': 0.50,
       'roi_bottom_y_ratio': 0.99,
       'hough_rho': 1,
       'hough_theta': np.pi / 180,
       'hough_threshold': 5,
       'hough_min_line_length': 15,
       'hough_max_line_gap': 300,
       'slope_threshold': 0.3,
       'ransac_iterations': 1000,
       'ransac_distance_threshold': 30,
       'ransac_min_inliers_ratio': 0.3,
   }

   result = detect_lanes(test_image, config=extreme_config, debug=True)
   ```

---

## Success Indicators

After running the fix, you should see:

✅ **Good signs:**
- `[DEBUG] Hough detected 30-150 lines` (not 0!)
- `[DEBUG] Left candidates: 15-50, Right candidates: 15-50`
- `✅ DETECTED` for at least one lane

❌ **Bad signs:**
- `[DEBUG] Hough detected 0 lines` - parameters still too restrictive
- `[DEBUG] Edge pixels in ROI: 0-100` - image quality issues or ROI misalignment

---

**This fix bypasses the kernel cache by defining the function directly. Run it and test immediately!**
