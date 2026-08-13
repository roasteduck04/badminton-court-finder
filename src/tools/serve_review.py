"""Local server for the court detection review page.

Serves local images and inference results — no external requests needed.

Run:  python src/tools/serve_review.py
Open: http://localhost:8001
"""

import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REVIEW_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "court_review.html")
IMAGES_DIR = os.path.join(ROOT, "data", "images")
RESULTS_FILE = os.path.join(ROOT, "data", "inference_results.json")


class ReviewHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/court_review.html":
            self._serve_file(REVIEW_HTML, "text/html")
            return

        if self.path == "/api/results":
            if os.path.isfile(RESULTS_FILE):
                self._serve_file(RESULTS_FILE, "application/json")
            else:
                self._json_response({"error": "No results. Run: python src/tools/run_inference.py"}, 404)
            return

        if self.path.startswith("/images/"):
            fname = unquote(self.path[len("/images/"):])
            fpath = os.path.join(IMAGES_DIR, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fpath)[1].lower()
                ct = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".bmp": "image/bmp",
                    ".webp": "image/webp",
                }.get(ext, "application/octet-stream")
                self._serve_file(fpath, ct)
                return

        self.send_error(404)

    def _serve_file(self, fpath, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", os.path.getsize(fpath))
        self.end_headers()
        with open(fpath, "rb") as f:
            self.wfile.write(f.read())

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        msg = str(args[0]) if args else ""
        if "/images/" not in msg:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    if not os.path.isfile(RESULTS_FILE):
        print(f"WARNING: No inference results found at {RESULTS_FILE}")
        print(f"  Run first: python src/tools/run_inference.py\n")

    port = 8001
    server = HTTPServer(("", port), ReviewHandler)
    url = f"http://localhost:{port}"
    print(f"Court Review running at {url}")
    print(f"  HTML:    {REVIEW_HTML}")
    print(f"  Images:  {IMAGES_DIR}")
    print(f"  Results: {RESULTS_FILE}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
