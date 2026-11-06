import os
import numpy as np
import cv2
import pytest
from src.tools import extract_frames as extract_frames_mod
from src.tools.extract_frames import extract_frames


@pytest.fixture
def sample_video(tmp_path):
    """Create a tiny synthetic video for testing."""
    video_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (320, 240))
    for i in range(90):  # 3 seconds at 30fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (i * 2, 100, 200)
        writer.write(frame)
    writer.release()
    return video_path


def test_extract_frames_basic(sample_video, tmp_path):
    output_dir = str(tmp_path / "frames")
    paths = extract_frames(sample_video, output_dir, fps=1.0)
    assert len(paths) == 3  # 3 seconds of video at 1 fps
    for p in paths:
        assert os.path.exists(p)
        img = cv2.imread(p)
        assert img is not None


def test_extract_frames_max_frames(sample_video, tmp_path):
    output_dir = str(tmp_path / "frames2")
    paths = extract_frames(sample_video, output_dir, fps=1.0, max_frames=2)
    assert len(paths) == 2


def test_extract_frames_higher_fps(sample_video, tmp_path):
    output_dir = str(tmp_path / "frames3")
    paths = extract_frames(sample_video, output_dir, fps=10.0)
    assert len(paths) >= 20  # ~30 frames at 10fps from 3s video


def test_extract_frames_every_n(sample_video, tmp_path):
    paths = extract_frames(sample_video, str(tmp_path / "frames4"), every_n=30)
    assert len(paths) == 3  # frames 0, 30, 60 of 90


@pytest.mark.parametrize("bad_fps", [0.0, float("nan")])
def test_extract_frames_missing_video_fps(sample_video, tmp_path, monkeypatch, bad_fps):
    """A container that reports 0/NaN FPS must not silently extract every frame."""
    real_capture = cv2.VideoCapture

    class NoFpsCapture:
        def __init__(self, path):
            self._cap = real_capture(path)

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return bad_fps
            return self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(extract_frames_mod.cv2, "VideoCapture", NoFpsCapture)
    out = tmp_path / "frames5"
    with pytest.raises(ValueError, match="every_n"):
        extract_frames(sample_video, str(out), fps=1.0)
    assert not out.exists()

    paths = extract_frames(sample_video, str(out), every_n=45)
    assert len(paths) == 2


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": -1}, {"every_n": 0}])
def test_extract_frames_invalid_args(sample_video, tmp_path, kwargs):
    with pytest.raises(ValueError):
        extract_frames(sample_video, str(tmp_path / "frames6"), **kwargs)