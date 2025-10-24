from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2
import sys
import os
import numpy as np

# -------- Parameters you can tweak --------
N_TO_SAVE = 50                 # Phase 1: how many frames to extract near the target
CLAHE_CLIP = 2.0               # Phase 2/3: CLAHE clip limit
CLAHE_GRID = (8, 8)            # Phase 2/3: CLAHE tile grid
AB_EPS = 2                     # Phase 3: ignore tiny chroma diffs (< AB_EPS)
EMA_ALPHA = 0.40               # Phase 3: temporal smoothing on masks (0..1); higher = faster adaptation
OPEN_KS = 3                    # Phase 3: morphology open kernel size (0 to disable)
CLOSE_KS = 7                   # Phase 3: morphology close kernel size (0 to disable)
MIN_BLOB_AREA = 150            # Phase 3: drop tiny blobs
MASK_COLOR = (0, 255, 255)     # Phase 4: overlay color (BGR) for mask
# -----------------------------------------

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

# ---------------------- Phase 1: frame extraction ----------------------

def extract_50_frames_at_trim_30pct(video_path, out_dir, n_save=N_TO_SAVE, prefix="frame"):
    """
    Extracts n_save frames starting at 30% into the middle 90% of the video.
    Saves PNGs and also returns the frames in-memory for Phase 2.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False, None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback if metadata missing
    if n_frames <= 0:
        tmp = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            tmp.append(f)
        cap.release()
        if not tmp:
            print("Error: Could not determine frame count or video is empty.")
            return False, None
        n_frames = len(tmp)
        cap = cv2.VideoCapture(video_path)

    # Trimmed window [5%, 95%]
    i_start = int((0.05 * n_frames) + 0.9999)  # ceil
    i_end   = int(0.95 * n_frames) - 1         # last valid index
    i_start = max(0, min(i_start, n_frames - 1))
    i_end   = max(0, min(i_end,   n_frames - 1))
    if i_end < i_start:
        i_start, i_end = 0, n_frames - 1

    span = i_end - i_start + 1

    # Target: 30% into the trimmed span
    i_target = i_start + round(0.30 * span)
    i_target = max(i_start, min(i_target, i_end))

    # Ensure we can extract n_save frames
    start_idx = min(i_target, max(i_start, i_end - n_save + 1))

    os.makedirs(out_dir, exist_ok=True)
    pad = max(6, len(str(n_frames)))

    # Seek and save sequentially + collect in memory
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    saved = 0
    current_idx = start_idx
    collected = []

    while saved < n_save and current_idx <= i_end:
        ok, frame = cap.read()
        if not ok:
            current_idx += 1
            continue
        out_path = os.path.join(out_dir, f"{prefix}_{current_idx:0{pad}d}.png")
        cv2.imwrite(out_path, frame)
        collected.append(frame)
        saved += 1
        current_idx += 1

    cap.release()

    if saved == 0:
        print("Error: No frames were saved.")
        return False, None

    print(
        f"[Phase 1] Saved {saved} frames to '{out_dir}'. "
        f"Trim window: [{i_start}, {i_end}] of {n_frames} total. "
        f"Target index: {i_target}, started at: {start_idx}"
    )
    return True, collected

# ---------------------- Phase 2: background / reference ----------------------

def normalize_lab_clahe(bgr, clip=CLAHE_CLIP, grid=CLAHE_GRID):
    """Normalize with Lab+CLAHE on L channel."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    Ln = clahe.apply(L)
    labn = cv2.merge([Ln, a, b])
    bgrn = cv2.cvtColor(labn, cv2.COLOR_Lab2BGR)
    return bgrn, (Ln, a, b)

def median_background(frames_bgr):
    """Per-pixel median background."""
    stack = np.stack(frames_bgr, axis=3)  # H x W x C x T
    med = np.median(stack, axis=3).astype(np.uint8)
    return med

def laplacian_sharpness(gray):
    """Variance of Laplacian as sharpness score (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def lab_ab_diff_mean(a1, b1, a2, b2, eps=AB_EPS):
    da = cv2.absdiff(a1, a2)
    db = cv2.absdiff(b1, b2)
    if eps > 0:
        da = np.where(da < eps, 0, da)
        db = np.where(db < eps, 0, db)
    return (da.mean() + db.mean()) / 2.0

def build_reference_from_frames(frames_bgr, out_dir):
    """
    Build a median background and pick the best single reference frame:
    - closest to the median in Lab a/b space
    - tie-broken by sharpness (Laplacian variance)
    Returns: ref_bgr_norm, (Lr, ar, br), B_med (median background, normalized too), ref_index
    """
    # Median background (raw)
    B = median_background(frames_bgr)

    # Normalize median background
    Bn, (LB, aB, bB) = normalize_lab_clahe(B)

    # Score each frame vs. median background in Lab a/b
    scores = []
    sharp = []
    for f in frames_bgr:
        fn, (Lf, af, bf) = normalize_lab_clahe(f)
        s = lab_ab_diff_mean(af, bf, aB, bB, eps=AB_EPS)
        scores.append(s)
        sharp.append(laplacian_sharpness(cv2.cvtColor(fn, cv2.COLOR_BGR2GRAY)))

    # Choose min score; tie-break by sharpness (pick higher sharpness)
    scores = np.array(scores)
    sharp = np.array(sharp)
    min_score = scores.min()
    candidates = np.where(np.isclose(scores, min_score, atol=1e-6))[0]
    if len(candidates) > 1:
        # among candidates, pick the sharpest
        idx = candidates[np.argmax(sharp[candidates])]
    else:
        idx = int(np.argmin(scores))

    ref_bgr = frames_bgr[idx]
    ref_bgr_norm, (Lr, ar, br) = normalize_lab_clahe(ref_bgr)

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "phase2_reference_frame_norm.png"), ref_bgr_norm)
    cv2.imwrite(os.path.join(out_dir, "phase2_background_median_norm.png"), Bn)

    print(f"[Phase 2] Picked reference frame index (within the 50): {idx}")
    return ref_bgr_norm, (Lr, ar, br), Bn, idx

# ---------------------- Phase 3: per-frame mask ----------------------

def postprocess_mask(mask_bin, open_ks=OPEN_KS, close_ks=CLOSE_KS, min_area=MIN_BLOB_AREA):
    out = mask_bin
    if open_ks and open_ks > 0:
        out = cv2.morphologyEx(
            out, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        )
    if close_ks and close_ks > 0:
        out = cv2.morphologyEx(
            out, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        )

    # remove small blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    cleaned = np.zeros_like(out)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned

def overlay_mask(frame_bgr, mask_bin, color=MASK_COLOR, alpha=0.45):
    color_layer = np.zeros_like(frame_bgr)
    color_layer[mask_bin > 0] = color
    return cv2.addWeighted(frame_bgr, 1.0, color_layer, alpha, 0)

def process_video_to_mask_and_overlay(video_path, ref_norm_lab, out_dir):
    """
    Phase 3 & 4:
    - For each frame in the video:
      normalize (same pipeline),
      compute Lab a/b difference vs reference,
      threshold, smooth temporally, morphology cleanup.
    - Write mask video and overlay video.
    """
    ref_bgr_norm, (Lr, ar, br) = ref_norm_lab

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0  # safe default if metadata is wrong

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_vid_path = os.path.join(out_dir, "phase4_masks.mp4")
    overlay_vid_path = os.path.join(out_dir, "phase4_overlay.mp4")
    vw_mask = cv2.VideoWriter(mask_vid_path, fourcc, fps, (w, h))
    vw_overlay = cv2.VideoWriter(overlay_vid_path, fourcc, fps, (w, h))

    ema = None  # temporal smoothing buffer

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Normalize frame identically to reference
        frame_norm, (Lf, af, bf) = normalize_lab_clahe(frame, clip=CLAHE_CLIP, grid=CLAHE_GRID)

        # Lab a/b difference (shadow-robust)
        da = cv2.absdiff(af, ar)
        db = cv2.absdiff(bf, br)
        if AB_EPS > 0:
            da[da < AB_EPS] = 0
            db[db < AB_EPS] = 0
        D = cv2.add(da, db)

        # Normalize to 0..255 for thresholding
        D8 = cv2.normalize(D, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Otsu threshold to binary
        _, mask = cv2.threshold(D8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Temporal EMA smoothing
        if ema is None:
            ema = mask.astype(np.float32)
        else:
            ema = EMA_ALPHA * mask.astype(np.float32) + (1.0 - EMA_ALPHA) * ema
        mask_smoothed = (ema > 127).astype(np.uint8) * 255

        # Morphology + small blob removal
        mask_clean = postprocess_mask(mask_smoothed, open_ks=OPEN_KS, close_ks=CLOSE_KS, min_area=MIN_BLOB_AREA)

        # Write mask video (as 3-channel for codec)
        mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        vw_mask.write(mask_bgr)

        # Overlay video for QA
        over = overlay_mask(frame, mask_clean, color=MASK_COLOR, alpha=0.45)
        vw_overlay.write(over)

        frame_idx += 1

    cap.release()
    vw_mask.release()
    vw_overlay.release()

    print(f"[Phase 4] Wrote mask video:   {mask_vid_path}")
    print(f"[Phase 4] Wrote overlay video:{overlay_vid_path}")
    return True

# ---------------------- Main ----------------------

if __name__ == "__main__":
    # Pick the video
    path = select_video_file()
    if not path:
        print("No file selected.")
        sys.exit(0)
    print("Selected:", path)

    # Pick the output folder
    out_folder = select_output_folder()
    if not out_folder:
        print("No output folder selected.")
        sys.exit(0)

    # Phase 1: extract & collect 50 frames around the target; save to disk too
    ok1, frames50 = extract_50_frames_at_trim_30pct(path, out_folder, n_save=N_TO_SAVE, prefix="grab")
    if not ok1 or not frames50:
        sys.exit(1)

    # Phase 2: build normalized reference and save artifacts
    ref_norm_bgr, (Lr, ar, br), Bn, ref_idx = build_reference_from_frames(frames50, out_folder)

    # Phase 3 & 4: run full-video mask generation and overlay export
    ok34 = process_video_to_mask_and_overlay(
        path,
        (ref_norm_bgr, (Lr, ar, br)),
        out_folder
    )
    if not ok34:
        sys.exit(1)

    print("Pipeline completed: Phase 1 → 2 → 3 → 4 ✅")
