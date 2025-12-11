"""
Diagnostic script to identify why lane detection is failing
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Load a sample image
train_dir = 'lane-detection-road-line-detection-image-dataset/Training'
image_paths = glob.glob(os.path.join(train_dir, '*.jpg')) + glob.glob(os.path.join(train_dir, '*.png'))

if len(image_paths) == 0:
    print("ERROR: No images found! Check dataset path.")
    exit(1)

# Load first image
img_path = image_paths[0]
print(f"Testing with image: {img_path}")
image = cv2.imread(img_path)

if image is None:
    print(f"ERROR: Could not load image from {img_path}")
    exit(1)

print(f"Original image shape: {image.shape}")

# Stage 1: Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"Grayscale shape: {gray.shape}")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

# Stage 2: Canny Edge Detection (TEST MULTIPLE THRESHOLDS)
print("\n" + "="*80)
print("TESTING CANNY EDGE DETECTION")
print("="*80)

thresholds = [
    (30, 90),
    (50, 150),
    (100, 200),
]

fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 5))

for idx, (low, high) in enumerate(thresholds):
    edges = cv2.Canny(blurred, low, high)
    edge_count = np.count_nonzero(edges)
    print(f"Canny ({low}, {high}): {edge_count} edge pixels ({edge_count / edges.size * 100:.2f}%)")

    axes[idx].imshow(edges, cmap='gray')
    axes[idx].set_title(f'Canny ({low}, {high})\n{edge_count} edge pixels')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('diagnostic_canny.png', dpi=150, bbox_inches='tight')
print(f"Saved: diagnostic_canny.png")
plt.close()

# Use middle threshold for remaining tests
edges = cv2.Canny(blurred, 50, 150)

# Stage 3: ROI Masking (TEST MULTIPLE ROI SIZES)
print("\n" + "="*80)
print("TESTING ROI CONFIGURATIONS")
print("="*80)

height, width = image.shape[:2]
print(f"Image dimensions: {width}x{height}")

roi_configs = [
    {"name": "Narrow", "bottom_w": 0.80, "top_w": 0.05, "top_y": 0.60, "bottom_y": 0.98},
    {"name": "Medium", "bottom_w": 0.95, "top_w": 0.08, "top_y": 0.60, "bottom_y": 0.98},
    {"name": "Wide", "bottom_w": 1.00, "top_w": 0.15, "top_y": 0.55, "bottom_y": 0.99},
]

fig, axes = plt.subplots(1, len(roi_configs), figsize=(18, 6))

for idx, cfg in enumerate(roi_configs):
    # Create ROI
    bottom_left = (int(width * (1 - cfg['bottom_w']) / 2), int(height * cfg['bottom_y']))
    bottom_right = (int(width * (1 + cfg['bottom_w']) / 2), int(height * cfg['bottom_y']))
    top_left = (int(width * (1 - cfg['top_w']) / 2), int(height * cfg['top_y']))
    top_right = (int(width * (1 + cfg['top_w']) / 2), int(height * cfg['top_y']))

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Create mask
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    edge_count = np.count_nonzero(masked_edges)
    print(f"ROI {cfg['name']}: {edge_count} edge pixels in ROI")
    print(f"  Vertices: BL={bottom_left}, TL={top_left}, TR={top_right}, BR={bottom_right}")

    # Visualize
    vis = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    cv2.polylines(vis, vertices, True, (0, 255, 0), 3)
    overlay = vis.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

    axes[idx].imshow(overlay)
    axes[idx].set_title(f'{cfg["name"]} ROI\n{edge_count} edge pixels')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('diagnostic_roi.png', dpi=150, bbox_inches='tight')
print(f"Saved: diagnostic_roi.png")
plt.close()

# Use medium ROI for Hough test
bottom_left = (int(width * 0.025), int(height * 0.98))
bottom_right = (int(width * 0.975), int(height * 0.98))
top_left = (int(width * 0.46), int(height * 0.60))
top_right = (int(width * 0.54), int(height * 0.60))

vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
mask = np.zeros_like(edges)
cv2.fillPoly(mask, vertices, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# Stage 4: Hough Transform (TEST MULTIPLE THRESHOLDS)
print("\n" + "="*80)
print("TESTING HOUGH TRANSFORM PARAMETERS")
print("="*80)

hough_thresholds = [10, 20, 30, 40, 50, 75, 100]

for threshold in hough_thresholds:
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=30,  # Reduced from 50
        maxLineGap=100     # Increased from 50
    )

    line_count = len(lines) if lines is not None else 0
    print(f"Hough threshold={threshold:3d}: {line_count:4d} lines detected")

# Test with best parameters
print("\n" + "="*80)
print("TESTING WITH OPTIMIZED PARAMETERS")
print("="*80)

lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,              # Finer resolution
    theta=np.pi / 180,
    threshold=15,       # Much lower threshold
    minLineLength=30,   # Shorter minimum
    maxLineGap=150      # Larger gap tolerance
)

if lines is not None:
    print(f"✓ DETECTED {len(lines)} lines with optimized parameters!")

    # Analyze slopes
    left_count = 0
    right_count = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:
                left_count += 1
            elif slope > 0.5:
                right_count += 1

    print(f"  Left lane candidates: {left_count}")
    print(f"  Right lane candidates: {right_count}")

    # Visualize
    vis = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Lines (n={len(lines)})\nLeft: {left_count}, Right: {right_count}')
    plt.axis('off')
    plt.savefig('diagnostic_hough_optimized.png', dpi=150, bbox_inches='tight')
    print(f"Saved: diagnostic_hough_optimized.png")
    plt.close()
else:
    print("✗ NO LINES DETECTED even with optimized parameters!")
    print("\nPossible issues:")
    print("  1. Image preprocessing is too aggressive")
    print("  2. ROI doesn't cover lane markings")
    print("  3. Lane markings are too faded in this image")
    print("\nRecommendation: Try a different image or adjust preprocessing")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print("\nCheck the generated images:")
print("  - diagnostic_canny.png")
print("  - diagnostic_roi.png")
print("  - diagnostic_hough_optimized.png (if lines detected)")
