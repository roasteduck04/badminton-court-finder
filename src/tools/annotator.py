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
    generate_line_mask,
    get_court_lines,
    compute_homography,
    project_points,
    validate_quadrilateral,
)

NUM_KEYPOINTS = 14
COURT_CLASS_NAMES = {0: "singles", 1: "doubles", 2: "alternative"}


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
        """Project all 14 template keypoints into normalized image space using
        the current homography. Returns None if a homography can't be computed.
        """
        H = self.get_homography()
        if H is None:
            return None
        return project_points(H, COURT_KEYPOINTS_TEMPLATE)

    def is_corner_quad_valid(self):
        """Check whether the 4 outer corner keypoints (K0-K3), if all visible,
        form a valid convex quadrilateral."""
        if not all(self.visibility[i] for i in range(4)):
            return False
        corners = np.array([self.keypoints[i] for i in range(4)])
        return validate_quadrilateral(corners)

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
# The GUI depends on tkinter/cv2/PIL at import time only when actually run,
# so that the logic above stays importable/testable in headless environments.


class CourtAnnotator:
    """Tkinter + OpenCV GUI for annotating court keypoints on frame images.

    Controls:
      - Left click on canvas: place/move the currently selected keypoint.
      - Right click on canvas: clear the currently selected keypoint.
      - Drag with left mouse button: move an existing keypoint under the cursor.
      - Number keys 0-9: select keypoint 0-9. Keypoints 10-13 selectable via
        the side panel listbox.
      - z: undo, y: redo, s: save annotation + mask, n: next frame, p: prev frame.
    """

    CANVAS_SIZE = 720
    POINT_RADIUS = 5
    NEAR_THRESHOLD_PX = 15

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
        self.image = None       # OpenCV BGR image, original resolution
        self.display_img = None  # PhotoImage currently shown
        self.scale = 1.0
        self.drag_kp = None

        self.root.title("CourtVisionNet Annotator")

        # Layout: canvas on the left, side panel on the right.
        main = tk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, bg="black")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        side = tk.Frame(main, width=260)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        self.frame_label = tk.Label(side, text="")
        self.frame_label.pack(anchor="w", padx=8, pady=(8, 0))

        self.kp_listbox = tk.Listbox(side, height=14)
        for i in range(NUM_KEYPOINTS):
            self.kp_listbox.insert(tk.END, f"K{i}")
        self.kp_listbox.select_set(0)
        self.kp_listbox.bind("<<ListboxSelect>>", self.on_kp_select)
        self.kp_listbox.pack(padx=8, pady=8, fill=tk.X)

        self.class_var = tk.IntVar(value=self.state.court_class)
        class_frame = tk.Frame(side)
        class_frame.pack(anchor="w", padx=8, pady=(0, 8))
        tk.Label(class_frame, text="Court class:").pack(side=tk.LEFT)
        for cls_id, name in COURT_CLASS_NAMES.items():
            tk.Radiobutton(
                class_frame, text=name, variable=self.class_var, value=cls_id,
                command=self.on_class_change,
            ).pack(side=tk.LEFT)

        self.status_label = tk.Label(side, text="", wraplength=240, justify=tk.LEFT)
        self.status_label.pack(anchor="w", padx=8, pady=8)

        btns = tk.Frame(side)
        btns.pack(padx=8, pady=8, fill=tk.X)
        tk.Button(btns, text="Prev (p)", command=self.prev_frame).grid(row=0, column=0, sticky="ew")
        tk.Button(btns, text="Next (n)", command=self.next_frame).grid(row=0, column=1, sticky="ew")
        tk.Button(btns, text="Undo (z)", command=self.undo).grid(row=1, column=0, sticky="ew")
        tk.Button(btns, text="Redo (y)", command=self.redo).grid(row=1, column=1, sticky="ew")
        tk.Button(btns, text="Save (s)", command=self.save).grid(row=2, column=0, columnspan=2, sticky="ew")

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

        self.frame_label.config(
            text=f"Frame {idx + 1}/{len(self.frame_paths)}: {os.path.basename(path)}"
        )
        self.render()

    def next_frame(self):
        self.load_frame(self.frame_idx + 1)

    def prev_frame(self):
        self.load_frame(self.frame_idx - 1)

    # -- rendering -------------------------------------------------------------

    def render(self):
        import cv2
        from PIL import Image, ImageTk

        h, w = self.image.shape[:2]
        self.scale = self.CANVAS_SIZE / max(h, w)
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        img = cv2.resize(self.image, (disp_w, disp_h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Overlay projected court lines if enough keypoints are placed.
        projected = self.state.project_full_template()
        if projected is not None:
            norm_pts = projected.copy()
            norm_pts[:, 0] /= w
            norm_pts[:, 1] /= h
            for start_idx, end_idx in get_court_lines():
                p1 = norm_pts[start_idx]
                p2 = norm_pts[end_idx]
                pt1 = (int(p1[0] * disp_w), int(p1[1] * disp_h))
                pt2 = (int(p2[0] * disp_w), int(p2[1] * disp_h))
                cv2.line(img_rgb, pt1, pt2, (0, 255, 0), 1)

        for i in range(NUM_KEYPOINTS):
            if not self.state.visibility[i]:
                continue
            x, y = self.state.keypoints[i]
            px, py = int(x * disp_w), int(y * disp_h)
            color = (255, 0, 0) if i == self.selected_kp else (0, 200, 255)
            cv2.circle(img_rgb, (px, py), self.POINT_RADIUS, color, -1)
            cv2.putText(img_rgb, str(i), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

        self._pil_img = Image.fromarray(img_rgb)
        self.display_img = ImageTk.PhotoImage(self._pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.display_img)

        self.status_label.config(
            text=(
                f"Selected: K{self.selected_kp}\n"
                f"Visible keypoints: {self.state.visible_count()}/{NUM_KEYPOINTS}\n"
                f"Corners valid: {self.state.is_corner_quad_valid()}"
            )
        )

    # -- canvas geometry helpers ------------------------------------------------

    def _canvas_to_norm(self, cx, cy):
        h, w = self.image.shape[:2]
        disp_w, disp_h = w * self.scale, h * self.scale
        return (cx / disp_w, cy / disp_h)

    def _norm_to_canvas(self, x, y):
        h, w = self.image.shape[:2]
        disp_w, disp_h = w * self.scale, h * self.scale
        return (x * disp_w, y * disp_h)

    def _find_nearby_keypoint(self, cx, cy):
        best_idx, best_dist = None, self.NEAR_THRESHOLD_PX
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
        nearby = self._find_nearby_keypoint(event.x, event.y)
        if nearby is not None:
            self.selected_kp = nearby
            self.kp_listbox.selection_clear(0, self.tk.END)
            self.kp_listbox.selection_set(nearby)
            self.drag_kp = nearby
            return
        x, y = self._canvas_to_norm(event.x, event.y)
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        self.state.set_keypoint(self.selected_kp, x, y)
        self.drag_kp = self.selected_kp
        self.render()

    def on_right_click(self, event):
        self.state.clear_keypoint(self.selected_kp)
        self.render()

    def on_drag(self, event):
        if self.drag_kp is None:
            return
        x, y = self._canvas_to_norm(event.x, event.y)
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        self.state.keypoints[self.drag_kp] = [x, y]
        self.state.visibility[self.drag_kp] = 1
        self.render()

    def on_release(self, event):
        self.drag_kp = None

    def on_kp_select(self, event):
        selection = self.kp_listbox.curselection()
        if selection:
            self.selected_kp = selection[0]
            self.render()

    def on_class_change(self):
        self.state.court_class = self.class_var.get()

    def on_key(self, event):
        if event.char.isdigit():
            self.selected_kp = int(event.char)
            self.kp_listbox.selection_clear(0, self.tk.END)
            self.kp_listbox.selection_set(self.selected_kp)
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

    # -- actions -----------------------------------------------------------------

    def undo(self):
        self.state.undo()
        self.render()

    def redo(self):
        self.state.redo()
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
            text=f"Saved: {os.path.basename(ann_path)}\nMask: {os.path.basename(self._mask_path_for(path))}"
        )


def main():
    parser = argparse.ArgumentParser(description="Court keypoint annotation tool")
    parser.add_argument("--input-dir", required=True, help="Directory of frame images to annotate")
    parser.add_argument("--output-dir", default=None, help="Directory to write annotation JSON files")
    args = parser.parse_args()

    import tkinter as tk

    root = tk.Tk()
    CourtAnnotator(root, args.input_dir, args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
