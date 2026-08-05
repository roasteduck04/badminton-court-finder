import os
import cv2


def extract_frames(video_path, output_dir, fps=1.0, max_frames=None):
    """Extract frames from a video at the specified FPS.

    Args:
        video_path: path to the video file
        output_dir: directory to save extracted frames
        fps: frames per second to extract
        max_frames: maximum number of frames to extract (None = all)

    Returns:
        list of saved frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(video_fps / fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    args = parser.parse_args()

    if os.path.isdir(args.video_path):
        for fname in sorted(os.listdir(args.video_path)):
            if fname.lower().endswith((".mov", ".mp4", ".avi", ".mkv")):
                vpath = os.path.join(args.video_path, fname)
                vname = os.path.splitext(fname)[0]
                out = os.path.join(args.output_dir, vname)
                paths = extract_frames(vpath, out, fps=args.fps, max_frames=args.max_frames)
                print(f"{fname}: extracted {len(paths)} frames")
    else:
        paths = extract_frames(args.video_path, args.output_dir, fps=args.fps, max_frames=args.max_frames)
        print(f"Extracted {len(paths)} frames")
