"""Local server for the CourtVisionNet annotator.

Run:  python serve_annotator.py
Open: http://localhost:8000
"""

import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(ROOT, "data", "images")
ANNOTATIONS_DIR = os.path.join(ROOT, "data", "annotations")


class AnnotatorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/annotator.html":
            fpath = os.path.join(ROOT, "src", "tools", "annotator.html")
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
                self.send_response(200)
                ext = os.path.splitext(fname)[1].lower()
                ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                      ".png": "image/png", ".bmp": "image/bmp",
                      ".webp": "image/webp"}.get(ext, "application/octet-stream")
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", os.path.getsize(fpath))
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
                return

        return super().do_GET()

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

        self.send_error(404)

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if "/images/" not in (args[0] if args else ""):
            super().log_message(format, *args)


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    port = 8000
    server = HTTPServer(("", port), AnnotatorHandler)
    url = f"http://localhost:{port}"
    print(f"Annotator running at {url}")
    print(f"  Images:      {IMAGES_DIR}")
    print(f"  Annotations: {ANNOTATIONS_DIR}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
