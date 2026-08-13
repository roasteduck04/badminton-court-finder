"""Local server for the CourtVisionNet annotator.

Run:  python serve_annotator.py
Open: http://localhost:8000
"""

import json
import os
import shutil
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(ROOT, "data", "images")
ANNOTATIONS_DIR = os.path.join(ROOT, "data", "annotations")
SCRAPED_DIR = os.path.join(ROOT, "data", "scraped")
CVN_DATASET_DIR = os.path.join(ROOT, "data", "cvn_dataset")
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


class AnnotatorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/annotator.html":
            fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotator.html")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", os.path.getsize(fpath))
            self.end_headers()
            with open(fpath, "rb") as f:
                self.wfile.write(f.read())
            return

        if self.path == "/api/images":
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            files = sorted(
                f for f in os.listdir(IMAGES_DIR)
                if os.path.splitext(f)[1].lower() in exts
            )
            self._json_response(files)
            return

        if self.path == "/api/annotations":
            anns = {}
            if os.path.isdir(ANNOTATIONS_DIR):
                for f in os.listdir(ANNOTATIONS_DIR):
                    if f.endswith(".json"):
                        with open(os.path.join(ANNOTATIONS_DIR, f)) as fh:
                            try:
                                anns[f] = json.load(fh)
                            except json.JSONDecodeError:
                                pass
            self._json_response(anns)
            return

        if self.path.startswith("/images/"):
            fname = unquote(self.path[len("/images/"):])
            fpath = os.path.join(IMAGES_DIR, fname)
            if os.path.isfile(fpath):
                self._serve_image(fpath)
                return

        if self.path == "/api/scraped":
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            files = []
            if os.path.isdir(SCRAPED_DIR):
                files = sorted(
                    f for f in os.listdir(SCRAPED_DIR)
                    if os.path.splitext(f)[1].lower() in exts
                )
            self._json_response({"files": files, "total": len(files)})
            return

        if self.path.startswith("/scraped/"):
            fname = unquote(self.path[len("/scraped/"):])
            fpath = os.path.join(SCRAPED_DIR, fname)
            if os.path.isfile(fpath):
                self._serve_image(fpath)
                return

        if self.path == "/dashboard":
            fpath = os.path.join(TOOLS_DIR, "dashboard.html")
            self._serve_file(fpath, "text/html")
            return

        if self.path == "/api/dashboard/sources":
            sources = self._get_dashboard_sources()
            self._json_response(sources)
            return

        if self.path.startswith("/api/dashboard/data/"):
            source = unquote(self.path[len("/api/dashboard/data/"):])
            data = self._get_source_data(source)
            self._json_response(data)
            return

        if self.path.startswith("/dataset/"):
            parts = unquote(self.path[len("/dataset/"):])
            fpath = os.path.join(CVN_DATASET_DIR, parts)
            if os.path.isfile(fpath):
                self._serve_image(fpath)
                return

        if self.path.startswith("/blender/"):
            fname = unquote(self.path[len("/blender/"):])
            fpath = os.path.join(ROOT, "data", "blender", "images", fname)
            if os.path.isfile(fpath):
                self._serve_image(fpath)
                return

        if self.path == "/favicon.ico":
            self.send_error(404)
            return

        return super().do_GET()

    def _serve_image(self, fpath):
        self.send_response(200)
        ext = os.path.splitext(fpath)[1].lower()
        ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".png": "image/png", ".bmp": "image/bmp",
              ".webp": "image/webp"}.get(ext, "application/octet-stream")
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", os.path.getsize(fpath))
        self.end_headers()
        with open(fpath, "rb") as f:
            self.wfile.write(f.read())

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            fname = body.get("filename", "")
            data = body.get("data", {})
            if not fname or not fname.endswith(".json"):
                self._json_response({"error": "invalid filename"}, 400)
                return
            os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
            fpath = os.path.join(ANNOTATIONS_DIR, fname)
            with open(fpath, "w") as f:
                json.dump(data, f, indent=2)
            self._json_response({"ok": True})
            return

        if self.path == "/api/scraped/keep":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            fname = body.get("filename", "")
            if not fname:
                self._json_response({"error": "missing filename"}, 400)
                return
            src = os.path.join(SCRAPED_DIR, fname)
            if not os.path.isfile(src):
                self._json_response({"error": "file not found"}, 404)
                return
            os.makedirs(IMAGES_DIR, exist_ok=True)
            shutil.move(src, os.path.join(IMAGES_DIR, fname))
            self._json_response({"ok": True})
            return

        if self.path == "/api/scraped/discard":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            fname = body.get("filename", "")
            if not fname:
                self._json_response({"error": "missing filename"}, 400)
                return
            fpath = os.path.join(SCRAPED_DIR, fname)
            if not os.path.isfile(fpath):
                self._json_response({"error": "file not found"}, 404)
                return
            os.remove(fpath)
            self._json_response({"ok": True})
            return

        self.send_error(404)

    def _serve_file(self, fpath, content_type):
        if not os.path.isfile(fpath):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", os.path.getsize(fpath))
        self.end_headers()
        with open(fpath, "rb") as f:
            self.wfile.write(f.read())

    def _get_dashboard_sources(self):
        sources = []
        if os.path.isdir(ANNOTATIONS_DIR):
            n = len([f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".json")])
            if n:
                sources.append({"id": "mine", "name": "My Annotations", "count": n})
        for split in ["train", "valid", "test"]:
            ann_dir = os.path.join(CVN_DATASET_DIR, split, "annotations")
            if os.path.isdir(ann_dir):
                n = len([f for f in os.listdir(ann_dir) if f.endswith(".json")])
                if n:
                    sources.append({"id": f"roboflow-{split}", "name": f"Roboflow {split}", "count": n})
        blender_dir = os.path.join(ROOT, "data", "blender")
        if os.path.isdir(blender_dir):
            ann_dir = os.path.join(blender_dir, "annotations")
            if os.path.isdir(ann_dir):
                n = len([f for f in os.listdir(ann_dir) if f.endswith(".json")])
                if n:
                    sources.append({"id": "blender", "name": "Blender Synthetic", "count": n})
        return sources

    def _get_source_data(self, source):
        if source == "mine":
            return self._load_cvn_annotations(ANNOTATIONS_DIR, IMAGES_DIR, "/images/")
        if source.startswith("roboflow-"):
            split = source[len("roboflow-"):]
            ann_dir = os.path.join(CVN_DATASET_DIR, split, "annotations")
            img_dir = os.path.join(CVN_DATASET_DIR, split, "images")
            return self._load_cvn_annotations(ann_dir, img_dir, f"/dataset/{split}/images/")
        if source == "blender":
            ann_dir = os.path.join(ROOT, "data", "blender", "annotations")
            img_dir = os.path.join(ROOT, "data", "blender", "images")
            return self._load_cvn_annotations(ann_dir, img_dir, "/blender/")
        return []

    def _load_cvn_annotations(self, ann_dir, img_dir, img_prefix):
        items = []
        if not os.path.isdir(ann_dir):
            return items
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(ann_dir, fname)) as f:
                try:
                    ann = json.load(f)
                except json.JSONDecodeError:
                    continue
            img_name = ann.get("image_path", fname.replace(".json", ".jpg"))
            vis = ann.get("visibility", [])
            n_vis = sum(1 for v in vis if v)
            items.append({
                "filename": img_name,
                "img_url": img_prefix + img_name,
                "keypoints": ann.get("keypoints", []),
                "visibility": vis,
                "n_visible": n_vis,
            })
        return items

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        msg = str(args[0]) if args else ""
        if "/images/" not in msg and "/scraped/" not in msg and "/dataset/" not in msg and "/blender/" not in msg:
            super().log_message(format, *args)


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(SCRAPED_DIR, exist_ok=True)

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    port = 8000
    server = ThreadedHTTPServer(("", port), AnnotatorHandler)
    url = f"http://localhost:{port}"
    print(f"Annotator running at {url}")
    print(f"  Images:      {IMAGES_DIR}")
    print(f"  Annotations: {ANNOTATIONS_DIR}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
