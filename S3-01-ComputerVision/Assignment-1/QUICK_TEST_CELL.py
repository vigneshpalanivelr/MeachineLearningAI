"""
QUICK TEST - Run this after the EMERGENCY_FIX_CELL to verify it works
"""

print("="*80)
print("QUICK TEST - Testing Fixed Pipeline")
print("="*80)

# Test on first image with debug enabled
test_img = sample_images[0]

print(f"\nTesting on image shape: {test_img.shape}")
print("\nRunning pipeline with FIXED parameters...")
print("-"*80)

result = detect_lanes(test_img, config=None, debug=True)

print("-"*80)
print("\n📊 RESULTS:")
print(f"  Left Lane:  {'✅ DETECTED' if result['left_line'] else '❌ NOT DETECTED'}")
if result['left_line']:
    m, b = result['left_line']
    print(f"    └─ Line: y = {m:.4f}x + {b:.2f}")
    print(f"    └─ Confidence: {result['left_confidence']:.1%}")

print(f"  Right Lane: {'✅ DETECTED' if result['right_line'] else '❌ NOT DETECTED'}")
if result['right_line']:
    m, b = result['right_line']
    print(f"    └─ Line: y = {m:.4f}x + {b:.2f}")
    print(f"    └─ Confidence: {result['right_confidence']:.1%}")

print("\n" + "="*80)

if result['left_line'] and result['right_line']:
    print("🎉 SUCCESS! Both lanes detected!")
    print("The fixed parameters are working!")
elif result['left_line'] or result['right_line']:
    print("⚠️  PARTIAL SUCCESS - One lane detected")
    print("This is better than 0%, but we can improve further")
else:
    print("❌ STILL FAILING - No lanes detected")
    print("The image might be extremely challenging")
    print("Let's visualize the pipeline stages...")

# Visualize pipeline stages
if result['debug_images']:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original image
    axes[0, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('0. Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Preprocessed
    axes[0, 1].imshow(result['debug_images']['1_preprocessed'], cmap='gray')
    axes[0, 1].set_title('1. Preprocessed (CLAHE + Blur)', fontsize=14)
    axes[0, 1].axis('off')

    # Edges
    edge_count = np.count_nonzero(result['debug_images']['2_edges'])
    axes[0, 2].imshow(result['debug_images']['2_edges'], cmap='gray')
    axes[0, 2].set_title(f'2. Canny Edges\n({edge_count} edge pixels)', fontsize=14)
    axes[0, 2].axis('off')

    # ROI mask visualization
    roi_mask_vis = cv2.cvtColor(test_img.copy(), cv2.COLOR_BGR2RGB)
    height, width = test_img.shape[:2]
    bottom_left = (int(width * 0.05), int(height * 0.98))
    bottom_right = (int(width * 0.95), int(height * 0.98))
    top_left = (int(width * 0.40), int(height * 0.55))
    top_right = (int(width * 0.60), int(height * 0.55))
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.polylines(roi_mask_vis, vertices, True, (0, 255, 0), 3)
    axes[1, 0].imshow(roi_mask_vis)
    axes[1, 0].set_title('3. ROI (Green)', fontsize=14)
    axes[1, 0].axis('off')

    # Masked edges
    roi_edge_count = np.count_nonzero(result['debug_images']['3_masked_edges'])
    axes[1, 1].imshow(result['debug_images']['3_masked_edges'], cmap='gray')
    axes[1, 1].set_title(f'4. Edges in ROI\n({roi_edge_count} pixels)', fontsize=14)
    axes[1, 1].axis('off')

    # Final result
    axes[1, 2].imshow(cv2.cvtColor(result['debug_images']['4_final'], cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('5. Final Result', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('quick_test_debug.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Debug visualization saved to: quick_test_debug.png")
    plt.show()

print("="*80)
