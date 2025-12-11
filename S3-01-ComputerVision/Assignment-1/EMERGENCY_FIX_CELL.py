"""
EMERGENCY FIX - Copy this entire cell and run it in your Jupyter notebook
This will override the detect_lanes function with fixed parameters
"""

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
            'ransac_distance_threshold': 25,  # FIXED: Was 20, now 25
            'ransac_min_inliers_ratio': 0.4,  # FIXED: Was 0.5, now 0.4
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

    # Create ROI mask
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
        if x2 - x1 == 0:  # Skip vertical lines
            continue
        slope = (y2 - y1) / (x2 - x1)

        if slope < -config['slope_threshold']:  # Left lane (negative slope)
            left_lines.append(line[0])
        elif slope > config['slope_threshold']:  # Right lane (positive slope)
            right_lines.append(line[0])

    if debug:
        print(f"[DEBUG] Left lane candidates: {len(left_lines)}, Right lane candidates: {len(right_lines)}")

    # Stage 6: RANSAC Fitting
    left_slope, left_intercept = None, None
    right_slope, right_intercept = None, None
    left_confidence, right_confidence = 0, 0

    # Helper function to convert lines to points
    def lines_to_points(lines):
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))
            # Add midpoint for long lines
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 100:
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                points.append((mid_x, mid_y))
        return points

    # Helper function for RANSAC
    def ransac_line_fit(points, max_iterations=1000, distance_threshold=20, min_inliers_ratio=0.5):
        if len(points) < 2:
            return None, None, [], []

        min_inliers = int(len(points) * min_inliers_ratio)
        best_slope = None
        best_intercept = None
        best_inliers = []
        best_inlier_count = 0

        for iteration in range(max_iterations):
            # Randomly select 2 points
            sample_points = random.sample(points, 2)
            (x1, y1), (x2, y2) = sample_points

            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Count inliers
            inliers = []
            for point in points:
                x, y = point
                predicted_y = slope * x + intercept
                distance = abs(y - predicted_y)

                if distance < distance_threshold:
                    inliers.append(point)

            inlier_count = len(inliers)
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_slope = slope
                best_intercept = intercept
                best_inliers = inliers

        if best_inlier_count < min_inliers:
            return None, None, [], []

        # Refine with least squares on inliers
        if len(best_inliers) >= 2:
            x_coords = np.array([p[0] for p in best_inliers])
            y_coords = np.array([p[1] for p in best_inliers])
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            refined_slope, refined_intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]
            best_slope = refined_slope
            best_intercept = refined_intercept

        outliers = [p for p in points if p not in best_inliers]

        return best_slope, best_intercept, best_inliers, outliers

    # Fit left lane
    if len(left_lines) > 0:
        left_points = lines_to_points(left_lines)
        left_slope, left_intercept, left_inliers, _ = ransac_line_fit(
            left_points,
            max_iterations=config['ransac_iterations'],
            distance_threshold=config['ransac_distance_threshold'],
            min_inliers_ratio=config['ransac_min_inliers_ratio']
        )
        if left_slope is not None:
            left_confidence = len(left_inliers) / len(left_points)
            if debug:
                print(f"[DEBUG] Left lane: slope={left_slope:.4f}, intercept={left_intercept:.2f}, confidence={left_confidence:.2%}")

    # Fit right lane
    if len(right_lines) > 0:
        right_points = lines_to_points(right_lines)
        right_slope, right_intercept, right_inliers, _ = ransac_line_fit(
            right_points,
            max_iterations=config['ransac_iterations'],
            distance_threshold=config['ransac_distance_threshold'],
            min_inliers_ratio=config['ransac_min_inliers_ratio']
        )
        if right_slope is not None:
            right_confidence = len(right_inliers) / len(right_points)
            if debug:
                print(f"[DEBUG] Right lane: slope={right_slope:.4f}, intercept={right_intercept:.2f}, confidence={right_confidence:.2%}")

    # Stage 7: Line Extrapolation and Drawing
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

    # Blend overlay with original
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

print("="*80)
print("✅ FIXED detect_lanes() FUNCTION LOADED!")
print("="*80)
print("\nKey Changes:")
print("  • hough_threshold: 50 → 10 (detect more lines)")
print("  • hough_min_line_length: 50 → 20 (accept shorter segments)")
print("  • hough_max_line_gap: 50 → 200 (connect dashed lanes)")
print("  • roi_top_width_ratio: 0.08 → 0.20 (much wider ROI)")
print("  • canny thresholds: (50,150) → (30,100) (more sensitive)")
print("\nNow re-run your test cells - detection should improve dramatically!")
print("="*80)
