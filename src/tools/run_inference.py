"""Run court keypoint inference on all images in data/images/.

Uses the Roboflow REST API with badminton-court-dataset/2.
Saves results to data/inference_results.json.

Usage:
    python src/tools/run_inference.py

Set ROBOFLOW_API_KEY env var or create a .env file with it.
"""

import base64
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = ROOT / "data" / "images"
OUTPUT_FILE = ROOT / "data" / "inference_results.json"
MODEL_ID = "badminton-court-dataset/2"
CONFIDENCE = 0.01
OVERLAP = 0.5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ROBOFLOW_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    print("ERROR: Set ROBOFLOW_API_KEY in environment or .env file")
    print("  Get your key from: https://app.roboflow.com/settings/api")
    sys.exit(1)


def infer(api_key, image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    url = (
        f"https://detect.roboflow.com/{MODEL_ID}"
        f"?api_key={api_key}"
        f"&confidence={CONFIDENCE}"
        f"&overlap={OVERLAP}"
    )
    req = Request(url, data=b64.encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    api_key = get_api_key()
    images = sorted(
        f for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    print(f"Running inference on {len(images)} images with {MODEL_ID}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print()

    results = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i:2d}/{len(images)}] {img_path.name} ... ", end="", flush=True)
        try:
            result = infer(api_key, img_path)
            n_preds = len(result.get("predictions", []))
            top_conf = 0
            total_kps = 0
            for p in result.get("predictions", []):
                top_conf = max(top_conf, p.get("confidence", 0))
                total_kps += len([k for k in p.get("keypoints", []) if k.get("confidence", 0) > 0.3])

            results.append({
                "filename": img_path.name,
                "image": result.get("image", {}),
                "predictions": result.get("predictions", []),
            })
            status = f"{n_preds} det, {total_kps} kp>0.3, top {top_conf:.0%}" if n_preds else "no detection"
            print(status)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "filename": img_path.name,
                "image": {},
                "predictions": [],
                "error": str(e),
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"model": MODEL_ID, "confidence": CONFIDENCE, "results": results}, f, indent=2)

    detected = sum(1 for r in results if r["predictions"])
    print(f"\nDone: {detected}/{len(results)} detected")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
