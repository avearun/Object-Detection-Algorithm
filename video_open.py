from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2
import sys
import os

N_TO_SAVE = 50  # how many frames to extract

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

def extract_50_frames_at_trim_30pct(video_path, out_dir, n_save=50, prefix="frame"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False

    # Try to get total frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Some containers don't report properly; fallback to counting
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

    # Define trimmed window [5%, 95%]
    # i_start = ceil(0.05 * n_frames), i_end = floor(0.95 * n_frames) - 1
    i_start = int((0.05 * n_frames) + 0.9999)  # ceil
    i_end   = int(0.95 * n_frames) - 1         # last valid index in the window

    # Clamp to valid range
    i_start = max(0, min(i_start, n_frames - 1))
    i_end   = max(0, min(i_end,   n_frames - 1))
    if i_end < i_start:  # ultra-short video fallback
        i_start, i_end = 0, n_frames - 1

    span = i_end - i_start + 1

    # Target is 30% into the trimmed span
    i_target = i_start + round(0.30 * span)
    i_target = max(i_start, min(i_target, i_end))

    # Ensure we can extract n_save frames from start; shift back if close to end
    start_idx = min(i_target, max(i_start, i_end - n_save + 1))

    # Prepare output
    os.makedirs(out_dir, exist_ok=True)
    pad = max(6, len(str(n_frames)))  # zero-pad for nice sorting

    # Seek and save sequentially
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    saved = 0
    current_idx = start_idx

    while saved < n_save and current_idx <= i_end:
        ok, frame = cap.read()
        if not ok:
            # Some codecs stop returning frames right at the end—advance and retry
            current_idx += 1
            continue
        out_path = os.path.join(out_dir, f"{prefix}_{current_idx:0{pad}d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1
        current_idx += 1

    cap.release()

    if saved == 0:
        print("Error: No frames were saved.")
        return False

    print(
        f"Saved {saved} frames to '{out_dir}'. "
        f"Trim window: [{i_start}, {i_end}] of {n_frames} total. "
        f"Target index: {i_target}, started at: {start_idx}"
    )
    return True

if __name__ == "__main__":
    # 1) Pick the video
    path = select_video_file()
    if not path:
        print("No file selected.")
        sys.exit(0)
    print("Selected:", path)

    # 2) Pick the output folder
    out_folder = select_output_folder()
    if not out_folder:
        print("No output folder selected.")
        sys.exit(0)

    # 3) Extract frames
    ok = extract_50_frames_at_trim_30pct(path, out_folder, n_save=N_TO_SAVE, prefix="grab")
    if not ok:
        sys.exit(1)
