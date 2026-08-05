import os
import tempfile
import numpy as np
import cv2
import pytest
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
