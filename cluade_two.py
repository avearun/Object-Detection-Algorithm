from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2
import numpy as np
import sys
import os
from scipy import stats

N_SAMPLES = 50  # number of random sample frames
BG_METHOD = "mode"  # "mode" or "median"

# Mask generation parameters
MOG2_HISTORY = 500  # Number of frames for MOG2 learning
MOG2_VAR_THRESHOLD = 16  # Lower = more sensitive to changes
MOG2_DETECT_SHADOWS = True  # Detect and mark shadows
MIN_CONTOUR_AREA = 100  # Minimum area to keep (remove small noise)

def select_video_file():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Video File",
        "",
        "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
    )
    return file_path if file_path else None

def select_output_folder():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    out_dir = QFileDialog.getExistingDirectory(None, "Select Output Folder", "")
    return out_dir if out_dir else None

def normalize_image_clahe(image):
    """
    Apply CLAHE normalization on LAB color space.
    Best for drone images with varying lighting conditions.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return normalized

def normalize_image_simple(image):
    """
    Simple and fast normalization using histogram stretching.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def compute_background_mode(images):
    """
    Compute pixel-wise mode across all images.
    For each pixel location, find the most frequent color value.
    """
    print(f"\nComputing background using MODE method from {len(images)} images...")
    
    if len(images) == 0:
        return None
    
    # Stack all images
    h, w, c = images[0].shape
    img_stack = np.array(images)  # shape: (n_images, h, w, c)
    
    # Compute mode for each pixel across all images
    background = np.zeros((h, w, c), dtype=np.uint8)
    
    for ch in range(c):
        channel_stack = img_stack[:, :, :, ch]  # (n_images, h, w)
        mode_result = stats.mode(channel_stack, axis=0, keepdims=False)
        background[:, :, ch] = mode_result.mode.astype(np.uint8)
    
    print("✓ Background computed using MODE")
    return background

def compute_background_median(images):
    """
    Compute pixel-wise median across all images.
    More robust to outliers than mean.
    """
    print(f"\nComputing background using MEDIAN method from {len(images)} images...")
    
    if len(images) == 0:
        return None
    
    # Stack all images and compute median
    img_stack = np.array(images)
    background = np.median(img_stack, axis=0).astype(np.uint8)
    
    print("✓ Background computed using MEDIAN")
    return background

def post_process_mask(mask, min_area=100):
    """
    Post-process mask to remove noise and small artifacts.
    """
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Opening: remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(mask)
    
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask_clean, [contour], -1, 255, -1)
    
    return mask_clean

def generate_mask_videos_mog2(video_path, out_dir, i_start, i_end, fps):
    """
    Generate mask video using MOG2 background subtraction with sequential reading.
    Fast and efficient - no feature detection or alignment needed.
    
    Args:
        video_path: Path to original video
        out_dir: Output directory
        i_start: Start frame index (5%)
        i_end: End frame index (95%)
        fps: Frame rate of original video
    """
    print(f"\n{'='*60}")
    print("GENERATING MASK VIDEOS - MOG2 METHOD")
    print(f"{'='*60}")
    print(f"Processing frames [{i_start}, {i_end}]")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for mask generation")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video writers
    mask_video_path = os.path.join(out_dir, "mask_video.mp4")
    overlay_video_path = os.path.join(out_dir, "overlay_video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height), False)
    overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height), True)
    
    if not mask_writer.isOpened() or not overlay_writer.isOpened():
        print("Error: Could not create video writers")
        cap.release()
        return False
    
    # Initialize MOG2 background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS
    )
    
    # Sequential reading for performance
    total_frames = i_end - i_start + 1
    processed = 0
    current_idx = 0
    
    print("\nProcessing video sequentially...")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Only process frames in trimmed window
        if current_idx >= i_start and current_idx <= i_end:
            
            # Simple normalization for consistent processing
            frame_norm = normalize_image_simple(frame)
            
            # Apply MOG2 background subtraction
            fg_mask = bg_subtractor.apply(frame_norm, learningRate=-1)
            
            # Remove shadow pixels (marked as 127 by MOG2 if detectShadows=True)
            if MOG2_DETECT_SHADOWS:
                fg_mask[fg_mask == 127] = 0
            
            # Post-process mask to remove noise
            fg_mask = post_process_mask(fg_mask, min_area=MIN_CONTOUR_AREA)
            
            # Create overlay (mask in green on original frame)
            overlay = frame.copy()
            overlay[fg_mask > 0] = [0, 255, 0]  # Green color for moving objects
            overlay = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Write frames
            mask_writer.write(fg_mask)
            overlay_writer.write(overlay)
            
            processed += 1
            
            # Progress indicator
            if processed % 100 == 0 or processed == total_frames:
                progress = (processed / total_frames) * 100
                print(f"Progress: {processed}/{total_frames} frames ({progress:.1f}%)")
        
        current_idx += 1
        
        # Early exit if we've processed all frames in window
        if current_idx > i_end:
            break
    
    # Release resources
    cap.release()
    mask_writer.release()
    overlay_writer.release()
    
    print(f"\n✓ Mask videos generated successfully!")
    print(f"  - Mask video: {mask_video_path}")
    print(f"  - Overlay video: {overlay_video_path}")
    print(f"  - Total frames processed: {processed}")
    print(f"  - Method: MOG2 Background Subtraction (Adaptive Learning)")
    
    return True

def extract_and_model_background(video_path, out_dir, n_samples=50, bg_method="mode"):
    """
    Main function for background modeling and mask generation from drone video.
    
    Steps:
    1. Find reference frame at 30% of trimmed window [5%, 95%]
    2. Sample n_samples random frames uniformly from trimmed window
    3. Normalize all frames using CLAHE
    4. Compute background using mode or median (no alignment needed)
    5. Generate mask videos using MOG2
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    # Get video properties
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if n_frames <= 0:
        tmp_count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            tmp_count += 1
        cap.release()
        if tmp_count == 0:
            print("Error: Could not determine frame count or video is empty.")
            return False
        n_frames = tmp_count
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        fps = 30.0  # Default fallback
    
    print(f"Total frames in video: {n_frames}")
    print(f"Frame rate: {fps:.2f} fps")
    
    # Define trimmed window [5%, 95%]
    i_start = int(0.05 * n_frames)
    i_end = int(0.95 * n_frames)
    i_start = max(0, min(i_start, n_frames - 1))
    i_end = max(0, min(i_end, n_frames - 1))
    
    print(f"Trimmed window: [{i_start}, {i_end}]")
    
    # Step 1: Get reference frame at 30% of trimmed window
    span = i_end - i_start + 1
    i_reference = i_start + int(0.30 * span)
    i_reference = max(i_start, min(i_reference, i_end))
    
    print(f"\n--- Step 1: Extracting Reference Frame ---")
    print(f"Reference frame index: {i_reference}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_reference)
    ok, ref_frame = cap.read()
    if not ok:
        print("Error: Could not read reference frame")
        cap.release()
        return False
    
    # Step 2: Sample random frames uniformly
    print(f"\n--- Step 2: Sampling {n_samples} Random Frames ---")
    np.random.seed(42)  # for reproducibility
    
    # Generate random indices excluding reference frame
    available_indices = list(range(i_start, i_end + 1))
    if i_reference in available_indices:
        available_indices.remove(i_reference)
    
    if len(available_indices) < n_samples:
        print(f"Warning: Only {len(available_indices)} frames available, using all")
        sample_indices = available_indices
    else:
        sample_indices = np.random.choice(available_indices, n_samples, replace=False)
        sample_indices = sorted(sample_indices)
    
    print(f"Sampled {len(sample_indices)} frame indices")
    
    # Read sample frames
    sample_frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            sample_frames.append((idx, frame))
    
    cap.release()
    print(f"Successfully read {len(sample_frames)} sample frames")
    
    # Create output directories
    os.makedirs(out_dir, exist_ok=True)
    normalized_dir = os.path.join(out_dir, "01_normalized")
    os.makedirs(normalized_dir, exist_ok=True)
    
    # Step 3: Normalize all images using CLAHE
    print(f"\n--- Step 3: Normalizing Images (CLAHE) ---")
    
    ref_normalized = normalize_image_clahe(ref_frame)
    cv2.imwrite(os.path.join(normalized_dir, f"reference_{i_reference:06d}.png"), ref_normalized)
    print(f"✓ Normalized reference frame")
    
    normalized_samples = []
    for idx, frame in sample_frames:
        normalized = normalize_image_clahe(frame)
        cv2.imwrite(os.path.join(normalized_dir, f"sample_{idx:06d}.png"), normalized)
        normalized_samples.append((idx, normalized))
    
    print(f"✓ Normalized {len(normalized_samples)} sample frames")
    
    # Step 4: Compute background directly from normalized frames (no alignment)
    print(f"\n--- Step 4: Computing Background (No Alignment) ---")
    
    # Collect all normalized frames for background computation
    all_normalized = [ref_normalized]
    all_normalized.extend([norm_frame for _, norm_frame in normalized_samples])
    
    print(f"Total frames for background modeling: {len(all_normalized)} (including reference)")
    
    if len(all_normalized) == 0:
        print("Error: No frames available for background modeling")
        return False
    
    if bg_method.lower() == "median":
        background = compute_background_median(all_normalized)
    else:  # default to mode
        background = compute_background_mode(all_normalized)
    
    if background is None:
        print("Error: Background computation failed")
        return False
    
    # Save background
    bg_path = os.path.join(out_dir, f"background_{bg_method}.png")
    cv2.imwrite(bg_path, background)
    
    print(f"\n{'='*60}")
    print(f"✓ BACKGROUND MODELING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {out_dir}")
    print(f"  - Normalized images: {normalized_dir}")
    print(f"  - Background image: {bg_path}")
    print(f"Method: {bg_method.upper()} (No Alignment)")
    print(f"Frames used: {len(all_normalized)}")
    
    # Step 5: Generate mask videos using MOG2
    print(f"\n--- Step 5: Generating Mask Videos (MOG2) ---")
    mask_ok = generate_mask_videos_mog2(video_path, out_dir, i_start, i_end, fps)
    
    if not mask_ok:
        print("Warning: Mask video generation failed")
        return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("DRONE BACKGROUND MODELING & MASK GENERATION TOOL")
    print("MOG2-Only Approach: Maximum Speed")
    print("="*60)
    
    # 1) Select video
    path = select_video_file()
    if not path:
        print("No file selected.")
        sys.exit(0)
    print(f"\nSelected video: {path}")
    
    # 2) Select output folder
    out_folder = select_output_folder()
    if not out_folder:
        print("No output folder selected.")
        sys.exit(0)
    print(f"Output folder: {out_folder}")
    
    # 3) Extract and model background + generate mask videos
    ok = extract_and_model_background(
        path, 
        out_folder, 
        n_samples=N_SAMPLES, 
        bg_method=BG_METHOD
    )
    
    if not ok:
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ ALL PROCESSES COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nApproach Summary:")
    print("  - Background Modeling: CLAHE + Mode/Median (No Alignment)")
    print("  - Mask Generation: MOG2 Adaptive Background Subtraction")
    print("  - Maximum Speed with Good Quality")
    sys.exit(0)