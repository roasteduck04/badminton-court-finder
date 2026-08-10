"""Annotation tool for CourtVisionNet.

Provides:
- `AnnotationState`: manages keypoints, visibility, undo/redo for a single frame.
- `save_annotation` / `load_annotation`: JSON persistence for a single frame's
  annotation.
- `CourtAnnotator`: a Tkinter + OpenCV GUI for clicking court keypoints on
  extracted frames, viewing the projected court overlay, and exporting
  annotations plus segmentation masks.

The GUI is launched via `python -m src.tools.annotator --input-dir <dir>`.
"""

import argparse
import copy
import json
import os

import numpy as np

from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    KEYPOINT_NAMES,
    generate_line_mask,
    get_court_lines,
    compute_homography,
    project_points,
    validate_quadrilateral,
)

from src.court_geometry import NUM_KEYPOINTS

COURT_CLASS_NAMES = {0: "singles", 1: "doubles", 2: "alternative"}

KEYPOINT_COLORS = (
    ["#FF0000"] * 4       # K0-K3:   back-L (red)
    + ["#FF8800"] * 4     # K4-K7:   long-svc-L (orange)
    + ["#22CC22"] * 5     # K8-K12:  short-svc-L (green)
    + ["#FFD700"] * 5     # K13-K17: net (gold)
    + ["#3388FF"] * 5     # K18-K22: short-svc-R (blue)
    + ["#CC44FF"] * 4     # K23-K26: long-svc-R (purple)
    + ["#FF4488"] * 4     # K27-K30: back-R (pink)
)


class AnnotationState:
    """Manages keypoint annotations for a single frame.

    Keypoints are stored as normalized [0, 1] image coordinates. An
    unplaced keypoint is represented as [-1.0, -1.0] with visibility 0.
    Every mutation is snapshotted so it can be undone/redone.
    """

    def __init__(self):
        self.keypoints = [[-1.0, -1.0] for _ in range(NUM_KEYPOINTS)]
        self.visibility = [0] * NUM_KEYPOINTS
        self.court_class = 1  # 0=singles, 1=doubles, 2=alternative
        self._history = []
        self._redo_stack = []

    def _snapshot(self):
        return {
            "keypoints": copy.deepcopy(self.keypoints),
            "visibility": list(self.visibility),
        }

    def _save_snapshot(self):
        self._history.append(self._snapshot())
        self._redo_stack.clear()

    def set_keypoint(self, idx, x, y):
        """Place/move keypoint `idx` to normalized coordinates (x, y) and mark visible."""
        self._save_snapshot()
        self.keypoints[idx] = [float(x), float(y)]
        self.visibility[idx] = 1

    def clear_keypoint(self, idx):
        """Remove keypoint `idx`, marking it not visible."""
        self._save_snapshot()
        self.keypoints[idx] = [-1.0, -1.0]
        self.visibility[idx] = 0

    def undo(self):
        if not self._history:
            return
        self._redo_stack.append(self._snapshot())
        snapshot = self._history.pop()
        self.keypoints = snapshot["keypoints"]
        self.visibility = snapshot["visibility"]

    def redo(self):
        if not self._redo_stack:
            return
        self._history.append(self._snapshot())
        snapshot = self._redo_stack.pop()
        self.keypoints = snapshot["keypoints"]
        self.visibility = snapshot["visibility"]

    def visible_count(self):
        return sum(self.visibility)

    def get_bounding_box(self):
        """Return (cx, cy, w, h) normalized bounding box of visible keypoints."""
        visible_pts = [
            self.keypoints[i] for i in range(NUM_KEYPOINTS) if self.visibility[i]
        ]
        if not visible_pts:
            return (0.0, 0.0, 0.0, 0.0)
        pts = np.array(visible_pts)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        return (float(cx), float(cy), float(w), float(h))

    def get_homography(self):
        """Compute a homography from the real-world court template to the
        current normalized-image keypoints, using only visible keypoints.

        Returns None if fewer than 4 keypoints are visible or the
        homography cannot be computed.
        """
        visible_idx = [i for i in range(NUM_KEYPOINTS) if self.visibility[i]]
        if len(visible_idx) < 4:
            return None
        src = COURT_KEYPOINTS_TEMPLATE[visible_idx]
        dst = np.array([self.keypoints[i] for i in visible_idx], dtype=np.float64)
        try:
            return compute_homography(src, dst)
        except ValueError:
            return None

    def project_full_template(self):
        """Project all template keypoints into normalized image space using
        the current homography. Returns None if a homography can't be computed.
        """
        H = self.get_homography()
        if H is None:
            return None
        return project_points(H, COURT_KEYPOINTS_TEMPLATE)

    # The 4 outer doubles corners in cyclic order (TL, TR, BR, BL)
    CORNER_INDICES = [0, 27, 30, 3]

    def is_corner_quad_valid(self):
        """Check whether the 4 outer corner keypoints (K0,K3,K27,K30), if
        all visible, form a valid convex quadrilateral."""
        if not all(self.visibility[i] for i in self.CORNER_INDICES):
            return False
        corners = np.array([self.keypoints[i] for i in self.CORNER_INDICES])
        return validate_quadrilateral(corners)

    def auto_sort_corners(self):
        """Sort the 4 outer corner keypoints by geometric position.

        Uses centroid-relative angles to assign:
        K0=top-left, K3=bottom-left, K27=top-right, K30=bottom-right.
        Only operates on corners that are currently visible.
        """
        ci = self.CORNER_INDICES
        if not all(self.visibility[i] for i in ci):
            return False

        self._save_snapshot()
        pts = [self.keypoints[i] for i in ci]
        arr = np.array(pts, dtype=np.float64)

        y_order = np.argsort(arr[:, 1])
        top_pair = arr[y_order[:2]]
        bottom_pair = arr[y_order[2:]]

        top_pair = top_pair[np.argsort(top_pair[:, 0])]
        bottom_pair = bottom_pair[np.argsort(bottom_pair[:, 0])]

        self.keypoints[0] = top_pair[0].tolist()       # top-left
        self.keypoints[27] = top_pair[1].tolist()      # top-right
        self.keypoints[30] = bottom_pair[1].tolist()   # bottom-right
        self.keypoints[3] = bottom_pair[0].tolist()    # bottom-left
        return True

    def auto_suggest_inner(self):
        """When 4 corners are placed, project the template to suggest
        positions for K4-K13. Only fills in keypoints not already placed.
        Returns the number of keypoints suggested.
        """
        if not all(self.visibility[i] for i in range(4)):
            return 0
        projected = self.project_full_template()
        if projected is None:
            return 0

        self._save_snapshot()
        count = 0
        for i in range(4, NUM_KEYPOINTS):
            if self.visibility[i]:
                continue
            x, y = projected[i]
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                self.keypoints[i] = [float(x), float(y)]
                self.visibility[i] = 1
                count += 1
        return count

    def to_dict(self, image_path="", image_size=(640, 640)):
        cx, cy, w, h = self.get_bounding_box()
        return {
            "image_path": image_path,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "court_class": self.court_class,
            "keypoints": [list(kp) for kp in self.keypoints],
            "visibility": list(self.visibility),
            "bounding_box": [cx, cy, w, h],
        }

    @classmethod
    def from_dict(cls, data):
        state = cls()
        state.keypoints = [list(kp) for kp in data["keypoints"]]
        state.visibility = list(data["visibility"])
        state.court_class = data.get("court_class", 1)
        return state


def save_annotation(state, image_path, output_path, image_size=(640, 640)):
    """Serialize `state` to JSON at `output_path`. Returns the dict written."""
    data = state.to_dict(image_path, image_size=image_size)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_annotation(path):
    """Load an `AnnotationState` from a JSON annotation file."""
    with open(path, "r") as f:
        data = json.load(f)
    return AnnotationState.from_dict(data)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class CourtAnnotator:
    """Tkinter + OpenCV GUI for annotating court keypoints on frame images.

    Controls:
      Mouse:
        Left click: place/move selected keypoint (or grab nearby point to drag)
        Right click: clear selected keypoint
        Scroll wheel: zoom in/out (centered on cursor)
        Middle click + drag (or Ctrl + left drag): pan the view

      Keyboard:
        0-9: select keypoint K0-K9
        z/y: undo/redo
        s: save annotation + mask
        n/p: next/prev frame
        a: auto-sort corners (K0-K3) by position
        g: auto-suggest inner keypoints from corners
        r: reset zoom to fit
        f: toggle fullscreen
    """

    CANVAS_SIZE = 800
    POINT_RADIUS = 3
    NEAR_THRESHOLD_PX = 12

    def __init__(self, root, input_dir, output_dir=None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.root = root
        self.input_dir = input_dir
        self.output_dir = output_dir or os.path.join(input_dir, "..", "annotations")
        self.output_dir = os.path.normpath(self.output_dir)
        self.mask_dir = os.path.normpath(os.path.join(self.output_dir, "..", "masks"))

        self.frame_paths = self._discover_frames(input_dir)
        if not self.frame_paths:
            raise ValueError(f"No image frames found in {input_dir}")

        self.frame_idx = 0
        self.selected_kp = 0
        self.state = AnnotationState()
        self.image = None
        self.display_img = None
        self.base_scale = 1.0
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_kp = None
        self.panning = False
        self.pan_start = None

        self.root.title("CourtVisionNet Annotator")
        self.root.configure(bg="#2b2b2b")

        main = tk.Frame(root, bg="#2b2b2b")
        main.pack(fill=tk.BOTH, expand=True)

        # Canvas
        canvas_frame = tk.Frame(main, bg="#1a1a1a")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="#1a1a1a", highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Button-2>", self.on_middle_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_release)
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Control-Button-1>", self.on_middle_click)
        self.canvas.bind("<Control-B1-Motion>", self.on_pan_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self.on_pan_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Side panel
        side = tk.Frame(main, width=300, bg="#333333")
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # Frame info
        self.frame_label = tk.Label(
            side, text="", bg="#333333", fg="white",
            font=("Segoe UI", 10, "bold"), anchor="w"
        )
        self.frame_label.pack(anchor="w", padx=8, pady=(8, 4), fill=tk.X)

        # Progress
        self.progress_label = tk.Label(
            side, text="", bg="#333333", fg="#aaaaaa",
            font=("Segoe UI", 9), anchor="w"
        )
        self.progress_label.pack(anchor="w", padx=8, pady=(0, 8), fill=tk.X)

        # Keypoint legend (scrollable)
        legend_label = tk.Label(
            side, text="Keypoints (click to select):", bg="#333333",
            fg="#cccccc", font=("Segoe UI", 9), anchor="w"
        )
        legend_label.pack(anchor="w", padx=8, pady=(4, 2))

        legend_frame = tk.Frame(side, bg="#333333")
        legend_frame.pack(padx=8, pady=(0, 8), fill=tk.X)

        self.kp_buttons = []
        for i in range(NUM_KEYPOINTS):
            btn = tk.Button(
                legend_frame,
                text=f"K{i}: {KEYPOINT_NAMES[i]}",
                bg="#444444", fg="white",
                activebackground="#555555", activeforeground="white",
                font=("Consolas", 8),
                anchor="w", relief=tk.FLAT, padx=4, pady=1,
                command=lambda idx=i: self._select_keypoint(idx),
            )
            btn.pack(fill=tk.X, pady=1)
            self.kp_buttons.append(btn)

        # Court class
        class_frame = tk.Frame(side, bg="#333333")
        class_frame.pack(anchor="w", padx=8, pady=(4, 4), fill=tk.X)
        tk.Label(class_frame, text="Court:", bg="#333333", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.class_var = tk.IntVar(value=1)
        for cls_id, name in COURT_CLASS_NAMES.items():
            tk.Radiobutton(
                class_frame, text=name, variable=self.class_var, value=cls_id,
                command=self.on_class_change, bg="#333333", fg="#cccccc",
                selectcolor="#555555", activebackground="#333333",
                font=("Segoe UI", 8),
            ).pack(side=tk.LEFT, padx=2)

        # Action buttons
        btn_frame = tk.Frame(side, bg="#333333")
        btn_frame.pack(padx=8, pady=4, fill=tk.X)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        btn_style = {"bg": "#555555", "fg": "white", "font": ("Segoe UI", 9),
                     "relief": tk.FLAT, "activebackground": "#666666"}

        tk.Button(btn_frame, text="< Prev (p)", command=self.prev_frame,
                  **btn_style).grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        tk.Button(btn_frame, text="Next (n) >", command=self.next_frame,
                  **btn_style).grid(row=0, column=1, sticky="ew", padx=1, pady=1)
        tk.Button(btn_frame, text="Undo (z)", command=self.undo,
                  **btn_style).grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        tk.Button(btn_frame, text="Redo (y)", command=self.redo,
                  **btn_style).grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        tk.Button(btn_frame, text="Sort Corners (a)", command=self.auto_sort,
                  **btn_style).grid(row=2, column=0, sticky="ew", padx=1, pady=1)
        tk.Button(btn_frame, text="Suggest Inner (g)", command=self.auto_suggest,
                  **btn_style).grid(row=2, column=1, sticky="ew", padx=1, pady=1)
        tk.Button(btn_frame, text="Reset Zoom (r)", command=self.reset_zoom,
                  **btn_style).grid(row=3, column=0, sticky="ew", padx=1, pady=1)
        tk.Button(btn_frame, text="Save (s)", command=self.save,
                  bg="#2d7d46", fg="white", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, activebackground="#3a9d5a",
                  ).grid(row=3, column=1, sticky="ew", padx=1, pady=1)

        # Status
        self.status_label = tk.Label(
            side, text="", wraplength=280, justify=tk.LEFT,
            bg="#333333", fg="#aaaaaa", font=("Segoe UI", 9), anchor="w"
        )
        self.status_label.pack(anchor="w", padx=8, pady=(8, 4), fill=tk.X)

        # Shortcuts reference
        shortcuts_text = (
            "Shortcuts:\n"
            "0-9: select K0-K9  |  Click legend: K10-K13\n"
            "Left click: place  |  Right click: clear\n"
            "Scroll: zoom  |  Mid/Ctrl+drag: pan\n"
            "z/y: undo/redo  |  s: save  |  n/p: next/prev\n"
            "a: sort corners  |  g: suggest inner  |  r: reset zoom"
        )
        tk.Label(
            side, text=shortcuts_text, bg="#2b2b2b", fg="#888888",
            font=("Consolas", 7), justify=tk.LEFT, anchor="w", padx=6, pady=4
        ).pack(side=tk.BOTTOM, fill=tk.X)

        root.bind("<Key>", self.on_key)

        self.load_frame(0)

    # -- frame discovery / IO -------------------------------------------------

    @staticmethod
    def _discover_frames(input_dir):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        return sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(exts)
        )

    def _annotation_path_for(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.output_dir, f"{stem}.json")

    def _mask_path_for(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.mask_dir, f"{stem}_mask.png")

    def load_frame(self, idx):
        import cv2

        if idx < 0 or idx >= len(self.frame_paths):
            return
        self.frame_idx = idx
        path = self.frame_paths[idx]
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"Could not read image: {path}")

        ann_path = self._annotation_path_for(path)
        if os.path.exists(ann_path):
            self.state = load_annotation(ann_path)
        else:
            self.state = AnnotationState()
        self.class_var.set(self.state.court_class)

        self.reset_zoom()

    def next_frame(self):
        self.load_frame(self.frame_idx + 1)

    def prev_frame(self):
        self.load_frame(self.frame_idx - 1)

    def _select_keypoint(self, idx):
        self.selected_kp = idx
        self.render()

    # -- zoom and pan ----------------------------------------------------------

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render()

    def _canvas_dims(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return self.CANVAS_SIZE, self.CANVAS_SIZE
        return w, h

    def _effective_scale(self):
        if self.image is None:
            return 1.0
        h, w = self.image.shape[:2]
        canvas_w, canvas_h = self._canvas_dims()
        self.base_scale = min(canvas_w, canvas_h) / max(h, w)
        return self.base_scale * self.zoom_level

    # -- rendering -------------------------------------------------------------

    def render(self):
        import cv2
        from PIL import Image, ImageTk

        if self.image is None:
            return

        scale = self._effective_scale()
        h, w = self.image.shape[:2]
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))

        img = cv2.resize(self.image, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        projected = self.state.project_full_template()
        if projected is not None:
            for start_idx, end_idx in get_court_lines():
                p1x = projected[start_idx][0] * disp_w
                p1y = projected[start_idx][1] * disp_h
                p2x = projected[end_idx][0] * disp_w
                p2y = projected[end_idx][1] * disp_h
                cv2.line(img_rgb, (int(p1x), int(p1y)), (int(p2x), int(p2y)),
                         (0, 255, 0), 1, cv2.LINE_AA)

        # Draw keypoints
        r = max(2, int(self.POINT_RADIUS * min(self.zoom_level, 2.0)))
        for i in range(NUM_KEYPOINTS):
            if not self.state.visibility[i]:
                continue
            x, y = self.state.keypoints[i]
            px, py = int(x * disp_w), int(y * disp_h)
            hex_color = KEYPOINT_COLORS[i]
            bgr = tuple(int(hex_color[j:j+2], 16) for j in (5, 3, 1))
            if i == self.selected_kp:
                cv2.circle(img_rgb, (px, py), r + 2, (255, 255, 255), 2, cv2.LINE_AA)
            rgb = (bgr[2], bgr[1], bgr[0])
            cv2.circle(img_rgb, (px, py), r, rgb, -1, cv2.LINE_AA)

            font_scale = 0.35 * min(self.zoom_level, 2.0)
            cv2.putText(img_rgb, str(i), (px + r + 2, py - r),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Apply pan offset — crop the rendered image to the canvas viewport
        canvas_w, canvas_h = self._canvas_dims()

        full_img = Image.fromarray(img_rgb)

        # Create canvas-sized image and paste the zoomed image with pan offset
        viewport = Image.new("RGB", (canvas_w, canvas_h), (26, 26, 26))
        paste_x = int((canvas_w - disp_w) / 2 + self.pan_x)
        paste_y = int((canvas_h - disp_h) / 2 + self.pan_y)
        viewport.paste(full_img, (paste_x, paste_y))

        self._paste_offset = (paste_x, paste_y)
        self.display_img = ImageTk.PhotoImage(viewport)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.display_img)

        # Update legend highlighting
        for i, btn in enumerate(self.kp_buttons):
            placed = "+" if self.state.visibility[i] else " "
            btn.config(text=f"{placed} K{i}: {KEYPOINT_NAMES[i]}")
            if i == self.selected_kp:
                btn.config(bg="#1a5276", fg="white")
            elif self.state.visibility[i]:
                btn.config(bg="#2d572c", fg="white")
            else:
                btn.config(bg="#444444", fg="#aaaaaa")

        # Update status
        path = self.frame_paths[self.frame_idx]
        self.frame_label.config(
            text=os.path.basename(path)
        )
        self.progress_label.config(
            text=f"Frame {self.frame_idx + 1}/{len(self.frame_paths)}  |  "
                 f"Keypoints: {self.state.visible_count()}/{NUM_KEYPOINTS}  |  "
                 f"Zoom: {self.zoom_level:.1f}x"
        )
        corners_ok = self.state.is_corner_quad_valid()
        self.status_label.config(
            text=f"Selected: K{self.selected_kp} ({KEYPOINT_NAMES[self.selected_kp]})\n"
                 f"Corners valid: {'Yes' if corners_ok else 'No (need K0-K3)'}"
        )

    # -- canvas geometry helpers ------------------------------------------------

    def _canvas_to_norm(self, cx, cy):
        """Convert canvas pixel coordinates to normalized [0,1] image coordinates."""
        scale = self._effective_scale()
        h, w = self.image.shape[:2]
        disp_w, disp_h = w * scale, h * scale
        canvas_w, canvas_h = self._canvas_dims()

        img_x = cx - (canvas_w - disp_w) / 2 - self.pan_x
        img_y = cy - (canvas_h - disp_h) / 2 - self.pan_y
        return (img_x / disp_w, img_y / disp_h)

    def _norm_to_canvas(self, x, y):
        """Convert normalized [0,1] image coordinates to canvas pixels."""
        scale = self._effective_scale()
        h, w = self.image.shape[:2]
        disp_w, disp_h = w * scale, h * scale
        canvas_w, canvas_h = self._canvas_dims()

        cx = x * disp_w + (canvas_w - disp_w) / 2 + self.pan_x
        cy = y * disp_h + (canvas_h - disp_h) / 2 + self.pan_y
        return (cx, cy)

    def _find_nearby_keypoint(self, cx, cy):
        threshold = self.NEAR_THRESHOLD_PX / max(self.zoom_level, 0.5)
        best_idx, best_dist = None, threshold
        for i in range(NUM_KEYPOINTS):
            if not self.state.visibility[i]:
                continue
            px, py = self._norm_to_canvas(*self.state.keypoints[i])
            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    # -- event handlers ---------------------------------------------------------

    def on_left_click(self, event):
        if self.panning:
            return
        nearby = self._find_nearby_keypoint(event.x, event.y)
        if nearby is not None:
            self.selected_kp = nearby
            self.drag_kp = nearby
            self.render()
            return
        x, y = self._canvas_to_norm(event.x, event.y)
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        self.state.set_keypoint(self.selected_kp, x, y)
        self.drag_kp = self.selected_kp
        # Auto-advance to next unplaced keypoint
        for offset in range(1, NUM_KEYPOINTS):
            next_kp = (self.selected_kp + offset) % NUM_KEYPOINTS
            if not self.state.visibility[next_kp]:
                self.selected_kp = next_kp
                break
        self.render()

    def on_right_click(self, event):
        nearby = self._find_nearby_keypoint(event.x, event.y)
        if nearby is not None:
            self.state.clear_keypoint(nearby)
        else:
            self.state.clear_keypoint(self.selected_kp)
        self.render()

    def on_middle_click(self, event):
        self.panning = True
        self.pan_start = (event.x, event.y)

    def on_pan_drag(self, event):
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self.pan_start = (event.x, event.y)
        self.render()

    def on_pan_release(self, event):
        self.panning = False
        self.pan_start = None

    def on_drag(self, event):
        if self.panning:
            self.on_pan_drag(event)
            return
        if self.drag_kp is None:
            return
        x, y = self._canvas_to_norm(event.x, event.y)
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        self.state.keypoints[self.drag_kp] = [x, y]
        self.state.visibility[self.drag_kp] = 1
        self.render()

    def on_release(self, event):
        if self.panning:
            self.on_pan_release(event)
            return
        self.drag_kp = None

    def on_scroll(self, event):
        old_zoom = self.zoom_level
        if event.delta > 0:
            self.zoom_level = min(self.zoom_level * 1.15, 10.0)
        else:
            self.zoom_level = max(self.zoom_level / 1.15, 0.5)

        # Zoom centered on cursor position
        zoom_ratio = self.zoom_level / old_zoom
        canvas_w, canvas_h = self._canvas_dims()
        cx = event.x - canvas_w / 2
        cy = event.y - canvas_h / 2
        self.pan_x = cx - zoom_ratio * (cx - self.pan_x)
        self.pan_y = cy - zoom_ratio * (cy - self.pan_y)

        self.render()

    def on_canvas_resize(self, event):
        self.render()

    def on_kp_select(self, event):
        selection = event.widget.curselection()
        if selection:
            self.selected_kp = selection[0]
            self.render()

    def on_class_change(self):
        self.state.court_class = self.class_var.get()

    def on_key(self, event):
        if event.char.isdigit():
            self.selected_kp = int(event.char)
            self.render()
        elif event.char == "z":
            self.undo()
        elif event.char == "y":
            self.redo()
        elif event.char == "s":
            self.save()
        elif event.char == "n":
            self.next_frame()
        elif event.char == "p":
            self.prev_frame()
        elif event.char == "a":
            self.auto_sort()
        elif event.char == "g":
            self.auto_suggest()
        elif event.char == "r":
            self.reset_zoom()

    # -- actions -----------------------------------------------------------------

    def undo(self):
        self.state.undo()
        self.render()

    def redo(self):
        self.state.redo()
        self.render()

    def auto_sort(self):
        if self.state.auto_sort_corners():
            self.status_label.config(text="Corners auto-sorted: K0=TL K1=TR K2=BR K3=BL")
        else:
            self.status_label.config(text="Need all 4 corners (K0-K3) placed to sort")
        self.render()

    def auto_suggest(self):
        count = self.state.auto_suggest_inner()
        if count > 0:
            self.status_label.config(text=f"Suggested {count} inner keypoints from corners")
        elif not all(self.state.visibility[i] for i in range(4)):
            self.status_label.config(text="Need all 4 corners placed first")
        else:
            self.status_label.config(text="All inner keypoints already placed")
        self.render()

    def save(self):
        import cv2

        path = self.frame_paths[self.frame_idx]
        h, w = self.image.shape[:2]
        ann_path = self._annotation_path_for(path)
        save_annotation(self.state, path, ann_path, image_size=(w, h))

        mask = generate_line_mask(
            np.array(self.state.keypoints), self.state.visibility, width=w, height=h
        )
        os.makedirs(self.mask_dir, exist_ok=True)
        cv2.imwrite(self._mask_path_for(path), mask)

        self.status_label.config(
            text=f"Saved: {os.path.basename(ann_path)}\n"
                 f"Mask: {os.path.basename(self._mask_path_for(path))}"
        )


def main():
    parser = argparse.ArgumentParser(description="Court keypoint annotation tool")
    parser.add_argument("--input-dir", required=True,
                        help="Directory of frame images to annotate")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write annotation JSON files")
    args = parser.parse_args()

    import tkinter as tk

    root = tk.Tk()
    root.geometry("1200x850")
    CourtAnnotator(root, args.input_dir, args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
