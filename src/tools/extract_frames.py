import math
import os

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int | None = None,
    every_n: int | None = None,
) -> list[str]:
    """Extract frames from a video at the specified FPS.

    Args:
        video_path: path to the video file
        output_dir: directory to save extracted frames
        fps: frames per second to extract (ignored when ``every_n`` is given)
        max_frames: maximum number of frames to extract (None = all)
        every_n: keep every n-th frame instead of sampling by time. Required when the
            container does not report a usable frame rate.

    Returns:
        list of saved frame file paths

    Raises:
        ValueError: if the video cannot be opened, the arguments are invalid, or the video
            reports no valid FPS and ``every_n`` was not given.
    """
    if every_n is not None and every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n}")
    if every_n is None and not (fps > 0 and math.isfinite(fps)):
        raise ValueError(f"fps must be a positive number, got {fps}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    if every_n is not None:
        frame_interval = every_n
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not (video_fps and math.isfinite(video_fps) and video_fps > 0):
            cap.release()
            raise ValueError(
                f"Video reports no valid frame rate (got {video_fps!r}): {video_path}. "
                "Pass every_n (CLI: --every-n N) to sample every N-th frame instead."
            )
        frame_interval = max(1, int(round(video_fps / fps)))

    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_paths = []
    frame_idx = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{video_name}_frame_{save_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)
            save_count += 1

            if max_frames is not None and save_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("video_path", help="Path to video file or directory of videos")
    parser.add_argument("output_dir", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--every-n", type=int, default=None,
                        help="Keep every N-th frame instead of sampling by --fps "
                             "(needed when the video reports no frame rate)")
    args = parser.parse_args()

    if os.path.isdir(args.video_path):
        for fname in sorted(os.listdir(args.video_path)):
            if fname.lower().endswith((".mov", ".mp4", ".avi", ".mkv")):
                vpath = os.path.join(args.video_path, fname)
                vname = os.path.splitext(fname)[0]
                out = os.path.join(args.output_dir, vname)
                paths = extract_frames(vpath, out, fps=args.fps, max_frames=args.max_frames,
                                       every_n=args.every_n)
                print(f"{fname}: extracted {len(paths)} frames")
    else:
        paths = extract_frames(args.video_path, args.output_dir, fps=args.fps,
                               max_frames=args.max_frames, every_n=args.every_n)
        print(f"Extracted {len(paths)} frames")
