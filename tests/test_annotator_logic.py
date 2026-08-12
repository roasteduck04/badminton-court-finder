import json
import numpy as np
import pytest
from src.tools.annotator import AnnotationState, save_annotation, load_annotation


def test_annotation_state_init():
    state = AnnotationState()
    assert len(state.keypoints) == 30
    assert all(v == 0 for v in state.visibility)
    assert state.court_class == 1  # doubles default


def test_set_keypoint():
    state = AnnotationState()
    state.set_keypoint(0, 0.5, 0.3)
    assert state.keypoints[0] == [0.5, 0.3]
    assert state.visibility[0] == 1


def test_clear_keypoint():
    state = AnnotationState()
    state.set_keypoint(0, 0.5, 0.3)
    state.clear_keypoint(0)
    assert state.keypoints[0] == [-1, -1]
    assert state.visibility[0] == 0


def test_undo_redo():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.2)
    state.set_keypoint(1, 0.9, 0.2)
    assert state.visibility[1] == 1

    state.undo()
    assert state.visibility[1] == 0
    assert state.visibility[0] == 1

    state.redo()
    assert state.visibility[1] == 1


def test_visible_count():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    assert state.visible_count() == 3


def test_save_and_load(tmp_path):
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    state.set_keypoint(3, 0.1, 0.9)
    state.court_class = 0  # singles

    out_path = str(tmp_path / "annotation.json")
    result = save_annotation(state, "test_frame.jpg", out_path)

    assert result["image_path"] == "test_frame.jpg"
    assert result["court_class"] == 0
    assert len(result["keypoints"]) == 30
    assert len(result["visibility"]) == 30
    assert result["visibility"][0] == 1
    assert result["visibility"][4] == 0

    loaded = load_annotation(out_path)
    assert loaded.keypoints[0] == [0.1, 0.1]
    assert loaded.visibility[0] == 1
    assert loaded.court_class == 0


def test_get_bounding_box():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    state.set_keypoint(3, 0.1, 0.9)
    cx, cy, w, h = state.get_bounding_box()
    assert abs(cx - 0.5) < 0.01
    assert abs(cy - 0.5) < 0.01
    assert abs(w - 0.8) < 0.01
    assert abs(h - 0.8) < 0.01


def test_get_homography_none_with_few_keypoints():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    assert state.get_homography() is None


def test_get_homography_and_project_full_template():
    state = AnnotationState()
    # Place the 4 outer corners (K0, K4, K25, K29).
    state.set_keypoint(0, 0.0, 0.0)
    state.set_keypoint(4, 0.0, 1.0)
    state.set_keypoint(25, 1.0, 0.0)
    state.set_keypoint(29, 1.0, 1.0)

    H = state.get_homography()
    assert H is not None
    assert H.shape == (3, 3)

    projected = state.project_full_template()
    assert projected is not None
    assert projected.shape == (30, 2)
    # K0 (real-world origin) should project back near normalized (0, 0).
    assert abs(projected[0, 0] - 0.0) < 1e-6
    assert abs(projected[0, 1] - 0.0) < 1e-6


def test_is_corner_quad_valid_true_for_rectangle():
    state = AnnotationState()
    # CORNER_INDICES = [0, 25, 29, 4] — TL, TR, BR, BL cyclic order
    state.set_keypoint(0, 0.1, 0.1)   # top-left
    state.set_keypoint(25, 0.9, 0.1)  # top-right
    state.set_keypoint(29, 0.9, 0.9)  # bottom-right
    state.set_keypoint(4, 0.1, 0.9)   # bottom-left
    assert state.is_corner_quad_valid() is True


def test_is_corner_quad_valid_false_when_missing_corner():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(4, 0.1, 0.9)
    state.set_keypoint(25, 0.9, 0.1)
    # K29 not placed
    assert state.is_corner_quad_valid() is False


def test_is_corner_quad_valid_false_for_self_intersecting():
    state = AnnotationState()
    # Bowtie ordering: swap positions to create self-intersection.
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(4, 0.9, 0.9)
    state.set_keypoint(25, 0.9, 0.1)
    state.set_keypoint(29, 0.1, 0.9)
    assert state.is_corner_quad_valid() is False
