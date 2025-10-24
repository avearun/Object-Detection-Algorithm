from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QCheckBox, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QDialog, QInputDialog
)
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import numpy as np
import sys
import os
from scipy import stats
from collections import deque
import time

# =========================================================
#                      CONFIG
# =========================================================
N_SAMPLES = 50  # number of random sample frames
BG_METHOD = "mode"  # "mode" or "median"

# Processing mode
PROCESSING_MODE = "high_quality"  # "fast" or "high_quality"

# Mask generation parameters - FAST mode
FAST_MOG2_HISTORY = 300
FAST_MOG2_VAR_THRESHOLD = 25
FAST_MIN_CONTOUR_AREA = 100

# Mask generation parameters - HIGH_QUALITY mode
HQ_MOG2_HISTORY = 500
HQ_MOG2_VAR_THRESHOLD = 16
HQ_MIN_CONTOUR_AREA = 30  # Lower threshold for small objects
HQ_TILE_SIZE = 512
HQ_TILE_OVERLAP = 64
HQ_TEMPORAL_FRAMES = 3  # Object must appear in N consecutive frames

MOG2_DETECT_SHADOWS = True

# Live preview toggle (split-screen)
ENABLE_LIVE_PREVIEW = True  # set False to disable preview window

# =========================================================
#                 COMMON UI HELPERS
# =========================================================
def cv_bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    if frame_bgr.ndim == 2:
        h, w = frame_bgr.shape
        qimg = QImage(frame_bgr.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    return qimg.copy()

# =========================================================
#           ROI EDITOR (Freehand + Polygon)
#           - Freehand brush/eraser, Undo/Redo
#           - Zoom (wheel), Pan (Space+drag)
#           - Overlay opacity
#           - Returns (polygon or None, binary mask or None)
# =========================================================
class ROIEditorDialog(QDialog):
    """
    ROI Editor with:
      - Modes: Freehand (brush) / Polygon
      - Brush size slider, Eraser toggle
      - Undo/Redo strokes
      - Zoom (wheel), Pan (Space+drag)
      - Overlay opacity
      - Returns: (polygon or None, binary mask or None)
    """
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Editor — Freehand/Polygon | zoom: wheel | pan: space+drag | undo: Ctrl+Z | redo: Ctrl+Y")
        self.frame = frame_bgr.copy()
        self.h, self.w = self.frame.shape[:2]

        # View state
        self.scale = 1.0
        self.tx, self.ty = 0.0, 0.0
        self.panning = False
        self.last_mouse_pos = QPoint()

        # ROI state
        self.mode_freehand = True   # True=Freehand, False=Polygon
        self.overlay_opacity = 0.35
        self.brush_size = 15
        self.eraser = False

        # Freehand: paint on mask in image coords
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.stroke_pts = []       # current stroke (list of QPoint)
        self.undo_stack = []       # list of dicts for undo/redo
        self.redo_stack = []

        # Polygon mode
        self.poly_points = []      # list of QPoint in screen space

        # Widgets
        self.view = QLabel()
        self.view.setMinimumSize(960, 540)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)

        self.btn_save = QPushButton("Save ROI")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_clear = QPushButton("Clear")

        self.chk_mode = QCheckBox("Freehand mode")
        self.chk_mode.setChecked(True)
        self.chk_eraser = QCheckBox("Eraser")
        self.chk_eraser.setChecked(False)

        self.slider_brush = QSlider(Qt.Horizontal); self.slider_brush.setRange(2, 50); self.slider_brush.setValue(self.brush_size)
        self.slider_opacity = QSlider(Qt.Horizontal); self.slider_opacity.setRange(0, 100); self.slider_opacity.setValue(int(self.overlay_opacity*100))

        top = QHBoxLayout()
        top.addWidget(QLabel("Mode:")); top.addWidget(self.chk_mode)
        top.addSpacing(10)
        top.addWidget(QLabel("Brush")); top.addWidget(self.slider_brush)
        top.addWidget(self.chk_eraser)
        top.addSpacing(10)
        top.addWidget(QLabel("Overlay")); top.addWidget(self.slider_opacity)
        top.addStretch(1)
        top.addWidget(self.btn_undo); top.addWidget(self.btn_redo)
        top.addWidget(self.btn_clear)
        top.addWidget(self.btn_cancel); top.addWidget(self.btn_save)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        layout.addLayout(top)

        # Signals
        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_undo.clicked.connect(self.on_undo)
        self.btn_redo.clicked.connect(self.on_redo)
        self.btn_clear.clicked.connect(self.on_clear)
        self.chk_mode.stateChanged.connect(self.on_mode_toggle)
        self.chk_eraser.stateChanged.connect(self.on_eraser)
        self.slider_brush.valueChanged.connect(self.on_brush)
        self.slider_opacity.valueChanged.connect(self.on_opacity)

        self.setFocusPolicy(Qt.StrongFocus)
        self.redraw()

    # ---------- helpers ----------
    def on_mode_toggle(self, _):
        self.mode_freehand = self.chk_mode.isChecked()
        self.redraw()

    def on_eraser(self, s): 
        self.eraser = (s == Qt.Checked)

    def on_brush(self, v): 
        self.brush_size = int(v)

    def on_opacity(self, v): 
        self.overlay_opacity = float(v)/100.0
        self.redraw()

    def on_undo(self):
        if not self.undo_stack: 
            return
        item = self.undo_stack.pop()
        self.redo_stack.append(item)
        kind = item["kind"]
        if kind == "paint":
            diff = item["diff"]
            self.mask = cv2.bitwise_xor(self.mask, diff)
        elif kind == "poly_point":
            if self.poly_points: 
                self.poly_points.pop()
        self.redraw()

    def on_redo(self):
        if not self.redo_stack: 
            return
        item = self.redo_stack.pop()
        self.undo_stack.append(item)
        kind = item["kind"]
        if kind == "paint":
            diff = item["diff"]
            self.mask = cv2.bitwise_xor(self.mask, diff)
        elif kind == "poly_point":
            self.poly_points.append(item["point"])
        self.redraw()

    def on_clear(self):
        self.mask[:] = 0
        self.poly_points.clear()
        self.stroke_pts.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.redraw()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space: 
            self.panning = True
        if (e.modifiers() & Qt.ControlModifier) and e.key() == Qt.Key_Z: 
            self.on_undo()
        if (e.modifiers() & Qt.ControlModifier) and e.key() == Qt.Key_Y: 
            self.on_redo()
        if e.key() == Qt.Key_C: 
            self.on_clear()
        if e.key() in (Qt.Key_Return, Qt.Key_Enter): 
            self.accept()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space: 
            self.panning = False

    # ---------- coordinate transforms ----------
    # screen = scale*img + translate -> img = (screen - translate)/scale
    def screen_to_img(self, x, y):
        xi = int(round((x - self.tx) / self.scale))
        yi = int(round((y - self.ty) / self.scale))
        return xi, yi

    # ---------- painting ----------
    def start_stroke(self, pos):
        self.stroke_pts = [pos]

    def paint_to(self, pos):
        if not self.stroke_pts: 
            self.stroke_pts = [pos]
            return
        p1 = self.stroke_pts[-1]; p2 = pos
        x1, y1 = self.screen_to_img(p1.x(), p1.y())
        x2, y2 = self.screen_to_img(p2.x(), p2.y())
        color = 0 if self.eraser else 255

        # draw onto temp diff to enable undo/redo via XOR
        diff = np.zeros_like(self.mask)
        cv2.line(diff, (x1, y1), (x2, y2), color, thickness=self.brush_size, lineType=cv2.LINE_AA)

        before = self.mask.copy()
        if self.eraser:
            self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(diff))
        else:
            self.mask = cv2.bitwise_or(self.mask, diff)

        delta = cv2.bitwise_xor(before, self.mask)

        if self.undo_stack and self.undo_stack[-1].get("in_progress", False):
            self.undo_stack[-1]["diff"] = cv2.bitwise_or(self.undo_stack[-1]["diff"], delta)
        else:
            self.undo_stack.append({"kind":"paint", "diff": delta, "in_progress": True})
        self.stroke_pts.append(pos)

    def end_stroke(self):
        if self.undo_stack and self.undo_stack[-1].get("in_progress", False):
            self.undo_stack[-1]["in_progress"] = False
            self.redo_stack.clear()

    # ---------- polygon mode ----------
    def add_poly_point(self, pos):
        self.poly_points.append(QPoint(pos.x(), pos.y()))
        self.undo_stack.append({"kind": "poly_point", "point": QPoint(pos.x(), pos.y())})
        self.redo_stack.clear()

    # ---------- rendering ----------
    def redraw(self):
        disp = self.render()
        self.view.setPixmap(QPixmap.fromImage(cv_bgr_to_qimage(disp)))

    def render(self):
        M = np.float32([[self.scale, 0, self.tx],
                        [0, self.scale, self.ty]])
        disp = cv2.warpAffine(self.frame, M,
                              (max(self.w, int(self.w*self.scale)+abs(int(self.tx))*2),
                               max(self.h, int(self.h*self.scale)+abs(int(self.ty))*2)))
        canvas = disp.copy()

        # warp mask for display
        mask_disp = cv2.warpAffine(self.mask, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST)
        if mask_disp.max() > 0:
            color = np.full_like(canvas, (255, 0, 0))
            blended = cv2.addWeighted(canvas, 1.0, color, float(self.overlay_opacity), 0.0)
            canvas = np.where(mask_disp[..., None] == 255, blended, canvas)

        # polygon overlay (if mode is polygon)
        if not self.mode_freehand and len(self.poly_points) > 0:
            for i in range(1, len(self.poly_points)):
                p1, p2 = self.poly_points[i-1], self.poly_points[i]
                cv2.line(canvas, (p1.x(), p1.y()), (p2.x(), p2.y()), (0,255,0), 2)
            for p in self.poly_points:
                cv2.circle(canvas, (p.x(), p.y()), 4, (0,0,255), -1)

        # help text
        tip = "Freehand: paint ROI (brush/eraser). Polygon: click points. Zoom wheel, Pan Space+Drag. Undo Ctrl+Z, Redo Ctrl+Y."
        cv2.rectangle(canvas, (0,0), (canvas.shape[1], 28), (0,0,0), -1)
        cv2.putText(canvas, tip, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return canvas

    # ---------- events ----------
    def eventFilter(self, obj, ev):
        if obj is self.view:
            if ev.type() == ev.MouseButtonPress:
                if ev.button() == Qt.LeftButton and not self.panning:
                    if self.mode_freehand:
                        self.start_stroke(ev.pos())
                    else:
                        self.add_poly_point(ev.pos())
                    self.redraw(); return True
                elif ev.button() == Qt.LeftButton and self.panning:
                    self.last_mouse_pos = ev.pos()
                    return True
            elif ev.type() == ev.MouseMove:
                if self.panning and (ev.buttons() & Qt.LeftButton):
                    delta = ev.pos() - self.last_mouse_pos
                    self.tx += delta.x(); self.ty += delta.y()
                    self.last_mouse_pos = ev.pos()
                    self.redraw(); return True
                if self.mode_freehand and (ev.buttons() & Qt.LeftButton):
                    self.paint_to(ev.pos())
                    self.redraw(); return True
            elif ev.type() == ev.MouseButtonRelease:
                if ev.button() == Qt.LeftButton and self.mode_freehand:
                    self.end_stroke(); return True
            elif ev.type() == ev.Wheel:
                angle = ev.angleDelta().y()
                factor = 1.1 if angle > 0 else (1/1.1)
                cursor = ev.pos()
                old = self.scale
                self.scale = float(np.clip(self.scale * factor, 0.2, 8.0))
                self.tx = cursor.x() - (cursor.x() - self.tx) * (self.scale / old)
                self.ty = cursor.y() - (cursor.y() - self.ty) * (self.scale / old)
                self.redraw(); return True
        return super().eventFilter(obj, ev)

    # ---------- export ----------
    def get_result(self):
        """
        Returns:
          (polygon_in_image_coords or None, binary_mask or None)
        Priority: if mask has content, return mask (more accurate).
                  else, return polygon (if >=3 points).
        """
        if self.mask.max() > 0:
            return None, self.mask.copy()

        if len(self.poly_points) >= 3:
            img_pts = []
            for p in self.poly_points:
                x = int(round((p.x() - self.tx) / self.scale))
                y = int(round((p.y() - self.ty) / self.scale))
                img_pts.append((x, y))
            poly = np.array(img_pts, dtype=np.int32)
            return poly, None

        return None, None

# =========================================================
#             LIVE PREVIEW (Split-screen Option B)
# =========================================================
class LivePreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Preview — Original | Mask | Overlay | Stats")
        self.resize(1200, 700)

        self.lbl_orig = QLabel("Original")
        self.lbl_mask = QLabel("Mask")
        self.lbl_overlay = QLabel("Overlay")
        for lbl in (self.lbl_orig, self.lbl_mask, self.lbl_overlay):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#222; color:#ccc; border:1px solid #444;")

        self.lbl_stats = QLabel("Stats")
        self.lbl_stats.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_stats.setStyleSheet("background:#111; color:#ddd; border:1px solid #333; padding:8px;")

        self.chk_pause = QCheckBox("Pause")
        self.chk_pause.setChecked(False)
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 4)  # 1x..4x (visual only)
        self.slider_speed.setValue(1)
        ctrl_box = QHBoxLayout()
        ctrl_box.addWidget(self.chk_pause)
        ctrl_box.addWidget(QLabel("Speed (visual):"))
        ctrl_box.addWidget(self.slider_speed)
        ctrl_box.addStretch(1)

        grid = QGridLayout(self)
        grid.addWidget(self.lbl_orig,    0, 0)
        grid.addWidget(self.lbl_mask,    0, 1)
        grid.addWidget(self.lbl_overlay, 1, 0)
        grid.addWidget(self.lbl_stats,   1, 1)
        grid.addLayout(ctrl_box, 2, 0, 1, 2)

        self.last_update = time.time()
        self.frame_counter = 0
        self.fps_est = 0.0

    def paused(self) -> bool:
        return self.chk_pause.isChecked()

    def update_views(self, orig_bgr, mask_gray, overlay_bgr, frame_idx, total, det_count):
        now = time.time()
        self.frame_counter += 1
        dt = now - self.last_update
        if dt >= 0.5:
            self.fps_est = self.frame_counter / dt
            self.frame_counter = 0
            self.last_update = now

        self.lbl_orig.setPixmap(QPixmap.fromImage(cv_bgr_to_qimage(orig_bgr)))
        if mask_gray.ndim == 3:
            mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
        self.lbl_mask.setPixmap(QPixmap.fromImage(cv_bgr_to_qimage(mask_gray)))
        self.lbl_overlay.setPixmap(QPixmap.fromImage(cv_bgr_to_qimage(overlay_bgr)))

        stats = [
            f"Frame: {frame_idx}/{total}",
            f"Detections (contours kept): {det_count}",
            f"UI FPS (approx): {self.fps_est:.1f}",
            f"Tip: Space to toggle pause (or checkbox)",
        ]
        self.lbl_stats.setText("\n".join(stats))
        QApplication.processEvents()

# =========================================================
#              ORIGINAL PIPELINE HELPERS (yours)
# =========================================================
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
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return normalized

def normalize_image_simple(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def compute_background_mode(images):
    print(f"\nComputing background using MODE method from {len(images)} images...")
    if len(images) == 0:
        return None
    h, w, c = images[0].shape
    img_stack = np.array(images)
    background = np.zeros((h, w, c), dtype=np.uint8)
    for ch in range(c):
        channel_stack = img_stack[:, :, :, ch]
        mode_result = stats.mode(channel_stack, axis=0, keepdims=False)
        background[:, :, ch] = mode_result.mode.astype(np.uint8)
    print("✓ Background computed using MODE")
    return background

def compute_background_median(images):
    print(f"\nComputing background using MEDIAN method from {len(images)} images...")
    if len(images) == 0:
        return None
    img_stack = np.array(images)
    background = np.median(img_stack, axis=0).astype(np.uint8)
    print("✓ Background computed using MEDIAN")
    return background

def post_process_mask(mask, min_area=100):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(mask)
    kept = 0
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask_clean, [contour], -1, 255, -1)
            kept += 1
    return mask_clean, kept

def create_roi_mask(shape, roi_polygon):
    if roi_polygon is None:
        return None
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    return mask

def process_tile_mog2(tile, mog2_instance, min_area):
    fg_mask = mog2_instance.apply(tile, learningRate=-1)
    if MOG2_DETECT_SHADOWS:
        fg_mask[fg_mask == 127] = 0
    fg_mask, kept = post_process_mask(fg_mask, min_area=min_area)
    return fg_mask, kept

def process_frame_tiled(frame, roi_mask, tile_size, overlap, mog2_instances, min_area):
    h, w = frame.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    kept_total = 0

    if roi_mask is None:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    stride = tile_size - overlap
    tiles_processed = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile_roi = roi_mask[y:y_end, x:x_end]
            if np.sum(tile_roi) == 0:
                continue

            tile = frame[y:y_end, x:x_end]
            tile_key = f"{x}_{y}"
            if tile_key not in mog2_instances:
                mog2_instances[tile_key] = cv2.createBackgroundSubtractorMOG2(
                    history=HQ_MOG2_HISTORY,
                    varThreshold=HQ_MOG2_VAR_THRESHOLD,
                    detectShadows=MOG2_DETECT_SHADOWS
                )

            tile_mask, kept = process_tile_mog2(tile, mog2_instances[tile_key], min_area)
            kept_total += kept
            final_mask[y:y_end, x:x_end] = np.maximum(final_mask[y:y_end, x:x_end], tile_mask)
            tiles_processed += 1

    return final_mask, tiles_processed, kept_total

def apply_temporal_consistency(mask_history, temporal_frames):
    if len(mask_history) < temporal_frames:
        return np.zeros_like(mask_history[0])
    recent_masks = np.array(list(mask_history)[-temporal_frames:])
    consistent_mask = np.all(recent_masks > 0, axis=0).astype(np.uint8) * 255
    return consistent_mask

# =========================================================
#         UPDATED: mask video generation (with preview)
#         NOTE: Now takes roi_mask (binary), not polygon
# =========================================================
def generate_mask_videos_hybrid(video_path, out_dir, i_start, i_end, fps, roi_mask, mode="fast"):
    """
    Hybrid approach with optional tiling (ROI) and live preview (split screen).
    roi_mask: np.uint8 binary mask (same size as frame) or None
    """
    print(f"\n{'='*60}")
    print(f"GENERATING MASK VIDEOS - HYBRID MODE ({mode.upper()})")
    print(f"{'='*60}")
    print(f"Processing frames [{i_start}, {i_end}]")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for mask generation")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if roi_mask is not None:
        # Ensure roi_mask size matches frame
        if roi_mask.shape[:2] != (height, width):
            roi_mask = cv2.resize(roi_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if mode == "fast":
        print("Mode: FAST - Global MOG2 processing")
        use_tiling = False
        mog2_history = FAST_MOG2_HISTORY
        mog2_var = FAST_MOG2_VAR_THRESHOLD
        min_area = FAST_MIN_CONTOUR_AREA
        use_temporal = False
    else:
        print("Mode: HIGH_QUALITY - Tiled processing in ROI + temporal consistency")
        use_tiling = True
        mog2_history = HQ_MOG2_HISTORY
        mog2_var = HQ_MOG2_VAR_THRESHOLD
        min_area = HQ_MIN_CONTOUR_AREA
        use_temporal = True
        if roi_mask is not None:
            print(f"  - Tile size: {HQ_TILE_SIZE}x{HQ_TILE_SIZE}, Overlap: {HQ_TILE_OVERLAP}px")
        print(f"  - Temporal consistency: {HQ_TEMPORAL_FRAMES} frames")

    # Outputs
    mask_video_path = os.path.join(out_dir, f"mask_video_{mode}.mp4")
    overlay_video_path = os.path.join(out_dir, f"overlay_video_{mode}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height), False)
    overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height), True)

    if not mask_writer.isOpened() or not overlay_writer.isOpened():
        print("Error: Could not create video writers")
        cap.release()
        return False

    # Subtractors
    if use_tiling and roi_mask is not None:
        mog2_instances = {}
        global_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history, varThreshold=mog2_var, detectShadows=MOG2_DETECT_SHADOWS
        )
    else:
        global_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history, varThreshold=mog2_var, detectShadows=MOG2_DETECT_SHADOWS
        )
        mog2_instances = None

    # Temporal
    if use_temporal:
        mask_history = deque(maxlen=HQ_TEMPORAL_FRAMES)

    total_frames = i_end - i_start + 1
    processed = 0
    current_idx = 0
    total_tiles = 0

    # Live preview
    preview = LivePreviewWindow() if ENABLE_LIVE_PREVIEW else None
    if preview:
        preview.show()

    print("\nProcessing video sequentially...")
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if current_idx >= i_start and current_idx <= i_end:
            frame_norm = normalize_image_simple(frame)

            if use_tiling and roi_mask is not None:
                # ROI via tiling
                roi_mask_tiled, tiles_proc, kept_tiles = process_frame_tiled(
                    frame_norm, roi_mask, HQ_TILE_SIZE, HQ_TILE_OVERLAP, 
                    mog2_instances, min_area
                )
                total_tiles += tiles_proc

                # outside ROI global
                outside_roi_mask = cv2.bitwise_not(roi_mask)
                outside_frame = cv2.bitwise_and(frame_norm, frame_norm, mask=outside_roi_mask)
                outside_fg = global_mog2.apply(outside_frame, learningRate=-1)
                if MOG2_DETECT_SHADOWS:
                    outside_fg[outside_fg == 127] = 0
                outside_fg, kept_out = post_process_mask(outside_fg, min_area=FAST_MIN_CONTOUR_AREA)

                fg_mask = cv2.bitwise_or(roi_mask_tiled, outside_fg)
                kept_count = kept_tiles + kept_out
            else:
                # global fast
                fg_mask = global_mog2.apply(frame_norm, learningRate=-1)
                if MOG2_DETECT_SHADOWS:
                    fg_mask[fg_mask == 127] = 0
                fg_mask, kept_count = post_process_mask(fg_mask, min_area=min_area)

            if use_temporal:
                mask_history.append(fg_mask.copy())
                fg_mask = apply_temporal_consistency(mask_history, HQ_TEMPORAL_FRAMES)

            # overlay
            overlay = frame.copy()
            overlay[fg_mask > 0] = [0, 255, 0]
            overlay = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # ROI boundary drawing from mask
            if roi_mask is not None:
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

            # write
            mask_writer.write(fg_mask)
            overlay_writer.write(overlay)

            processed += 1
            if processed % 100 == 0 or processed == total_frames:
                progress = (processed / total_frames) * 100
                if use_tiling and total_tiles > 0:
                    avg_tiles = total_tiles / processed
                    print(f"Progress: {processed}/{total_frames} ({progress:.1f}%) - Avg tiles/frame: {avg_tiles:.1f}")
                else:
                    print(f"Progress: {processed}/{total_frames} ({progress:.1f}%)")

            # live preview
            if preview:
                # allow pause
                while preview.paused():
                    QApplication.processEvents()
                    time.sleep(0.03)
                preview.update_views(frame, fg_mask, overlay, processed, total_frames, kept_count)

            # small sleep based on speed slider (visual only)
            if preview:
                sp = preview.slider_speed.value()  # 1..4
                time.sleep(0.001 * max(1, 5 - sp))

        current_idx += 1
        if current_idx > i_end:
            break

    cap.release()
    mask_writer.release()
    overlay_writer.release()

    t1 = time.time()
    print(f"\n✓ Mask videos generated successfully!")
    print(f"  - Mask video: {mask_video_path}")
    print(f"  - Overlay video: {overlay_video_path}")
    print(f"  - Mode: {mode.upper()}")
    print(f"  - Frames processed: {processed}")
    if use_tiling and total_tiles > 0:
        print(f"  - Total tiles processed: {total_tiles}")

    # Report ROI using mask
    if roi_mask is not None:
        nz = int((roi_mask > 0).sum())
        pct = 100.0 * nz / (height * width)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  - ROI mask: {nz} pixels ({pct:.2f}% of frame), {len(contours)} region(s)")
    else:
        print("  - ROI: none")

    print(f"  - Elapsed: {(t1 - t0):.2f}s")

    return True

# =========================================================
#         MAIN DRIVER (with ROI editor integration)
# =========================================================
def extract_and_model_background(video_path, out_dir, n_samples=50, bg_method="mode", processing_mode="fast"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False

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
        fps = 30.0

    print(f"Total frames in video: {n_frames}")
    print(f"Frame rate: {fps:.2f} fps")

    i_start = int(0.05 * n_frames)
    i_end = int(0.95 * n_frames)
    i_start = max(0, min(i_start, n_frames - 1))
    i_end = max(0, min(i_end, n_frames - 1))

    print(f"Trimmed window: [{i_start}, {i_end}]")

    # reference frame
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

    # Step 2: Sample random frames
    print(f"\n--- Step 2: Sampling {n_samples} Random Frames ---")
    np.random.seed(42)
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

    # Step 3: Normalize
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

    # Step 4: Compute background (mode/median) — kept for continuity (saved artifact)
    print(f"\n--- Step 4: Computing Background ---")
    all_normalized = [ref_normalized]
    all_normalized.extend([norm_frame for _, norm_frame in normalized_samples])
    print(f"Total frames for background modeling: {len(all_normalized)}")

    if bg_method.lower() == "median":
        background = compute_background_median(all_normalized)
    else:
        background = compute_background_mode(all_normalized)

    if background is None:
        print("Error: Background computation failed")
        return False

    bg_path = os.path.join(out_dir, f"background_{bg_method}.png")
    cv2.imwrite(bg_path, background)

    print(f"\n{'='*60}")
    print(f"✓ BACKGROUND MODELING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {out_dir}")
    print(f"  - Normalized images: {normalized_dir}")
    print(f"  - Background image: {bg_path}")

    # Step 5: ROI Selection (high_quality only) — NEW editor dialog (mask-first)
    roi_polygon = None
    roi_mask_binary = None
    if processing_mode == "high_quality":
        editor = ROIEditorDialog(ref_frame)
        if editor.exec_() == QDialog.Accepted:
            roi_poly, roi_mask = editor.get_result()
            if roi_mask is not None and roi_mask.any():
                roi_mask_binary = roi_mask
                roi_polygon = None
                print(f"✓ ROI mask drawn (pixels on: {int((roi_mask_binary>0).sum())})")
            elif roi_poly is not None and len(roi_poly) >= 3:
                roi_polygon = roi_poly
                roi_mask_binary = create_roi_mask(ref_frame.shape, roi_polygon)
                print(f"✓ ROI polygon defined with {len(roi_polygon)} points")
            else:
                print("No ROI provided — processing entire frame.")
        else:
            print("ROI selection canceled — processing entire frame.")

    # Step 6: Generate mask videos (with live preview)
    print(f"\n--- Step 5: Generating Mask Videos ---")
    mask_ok = generate_mask_videos_hybrid(
        video_path, out_dir, i_start, i_end, fps,
        roi_mask_binary,   # pass mask directly; None means full frame
        mode=processing_mode
    )

    if not mask_ok:
        print("Warning: Mask video generation failed")
        return False

    return True

# =========================================================
#                       MAIN
# =========================================================
if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    print("="*60)
    print("DRONE BACKGROUND MODELING & MASK GENERATION")
    print("Hybrid Tile + ROI Processing + Live Preview + ROI Editor")
    print("="*60)
    print(f"\nCurrent mode: {PROCESSING_MODE.upper()}")
    print("  - FAST: Global MOG2, no ROI, faster processing")
    print("  - HIGH_QUALITY: Tile-based in ROI, temporal consistency")
    print("\nTo change mode, edit PROCESSING_MODE near the top of the file.")

    # (Optional) quick runtime mode picker:
    # mode, ok = QInputDialog.getItem(None, "Select Mode", "Processing mode:",
    #                                 ["fast", "high_quality"], 1 if PROCESSING_MODE=="high_quality" else 0, False)
    # if ok and mode:
    #     PROCESSING_MODE = mode

    path = select_video_file()
    if not path:
        print("No file selected.")
        sys.exit(0)
    print(f"\nSelected video: {path}")

    out_folder = select_output_folder()
    if not out_folder:
        print("No output folder selected.")
        sys.exit(0)
    print(f"Output folder: {out_folder}")

    ok = extract_and_model_background(
        path,
        out_folder,
        n_samples=N_SAMPLES,
        bg_method=BG_METHOD,
        processing_mode=PROCESSING_MODE
    )

    if not ok:
        sys.exit(1)

    print(f"\n{'='*60}")
    print("✓ ALL PROCESSES COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    sys.exit(0)
