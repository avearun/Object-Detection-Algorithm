from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2
import numpy as np
import sys
import os
from scipy import stats

N_SAMPLES = 50  # number of random sample frames
BG_METHOD = "mode"  # "mode" or "median"

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

def align_image_to_reference(img, ref_img, min_match_count=10):
    """
    Align img to ref_img using SIFT features and homography with RANSAC.
    Returns aligned image or None if alignment fails.
    """
    # Convert to grayscale for feature detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Detect SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_img, None)
    
    if des1 is None or des2 is None:
        print("  ⚠ No features detected")
        return None
    
    # Match features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < min_match_count:
        print(f"  ⚠ Not enough matches: {len(good_matches)}/{min_match_count}")
        return None
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("  ⚠ Homography computation failed")
        return None
    
    # Warp image to align with reference
    h, w = ref_img.shape[:2]
    aligned = cv2.warpPerspective(img, H, (w, h))
    
    inliers = np.sum(mask)
    print(f"  ✓ Aligned: {len(good_matches)} matches, {inliers} inliers")
    
    return aligned

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
    # Mode is computed independently for each channel
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

def extract_and_model_background(video_path, out_dir, n_samples=50, bg_method="mode"):
    """
    Main function for background modeling from drone video.
    
    Steps:
    1. Find reference frame at 30% of trimmed window [5%, 95%]
    2. Sample n_samples random frames uniformly from trimmed window
    3. Normalize all frames using CLAHE
    4. Align sample frames to reference using homography
    5. Compute background using mode or median
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    # Get total frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    
    print(f"Total frames in video: {n_frames}")
    
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
    aligned_dir = os.path.join(out_dir, "02_aligned")
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    
    # Step 3: Normalize all images
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
    
    # Step 4: Align sample frames to reference
    print(f"\n--- Step 4: Aligning Frames to Reference (Homography/RANSAC) ---")
    
    aligned_frames = [ref_normalized]  # Reference is already aligned to itself
    cv2.imwrite(os.path.join(aligned_dir, f"reference_{i_reference:06d}.png"), ref_normalized)
    
    successful_alignments = 0
    for idx, norm_frame in normalized_samples:
        print(f"Aligning frame {idx}...", end="")
        aligned = align_image_to_reference(norm_frame, ref_normalized)
        
        if aligned is not None:
            aligned_frames.append(aligned)
            cv2.imwrite(os.path.join(aligned_dir, f"aligned_{idx:06d}.png"), aligned)
            successful_alignments += 1
        else:
            print(f"  ✗ Skipping frame {idx} (alignment failed)")
    
    print(f"\n✓ Successfully aligned {successful_alignments}/{len(normalized_samples)} frames")
    print(f"Total frames for background modeling: {len(aligned_frames)} (including reference)")
    
    if len(aligned_frames) == 0:
        print("Error: No frames available for background modeling")
        return False
    
    # Step 5: Compute background
    print(f"\n--- Step 5: Computing Background ---")
    
    if bg_method.lower() == "median":
        background = compute_background_median(aligned_frames)
    else:  # default to mode
        background = compute_background_mode(aligned_frames)
    
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
    print(f"  - Aligned images: {aligned_dir}")
    print(f"  - Background image: {bg_path}")
    print(f"Method: {bg_method.upper()}")
    print(f"Frames used: {len(aligned_frames)}")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("DRONE BACKGROUND MODELING TOOL")
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
    
    # 3) Extract and model background
    ok = extract_and_model_background(
        path, 
        out_folder, 
        n_samples=N_SAMPLES, 
        bg_method=BG_METHOD
    )
    
    if not ok:
        sys.exit(1)
    
    print("\n✓ Process completed successfully!")
    sys.exit(0)