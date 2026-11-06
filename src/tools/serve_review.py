"""Local server for the court detection review page.

Serves local images and inference results — no external requests needed.

Run:  python src/tools/serve_review.py [--host 127.0.0.1] [--port 8001] [--no-browser]
Open: http://127.0.0.1:8001

Binds to loopback by default, serves only explicit routes, and confines image access to the
images directory (see serve_annotator.py for the shared security model).
"""

import argparse
import json
import os
import shutil
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit

try:  # package import (tests, python -m src.tools.serve_review)
    from src.tools.serve_annotator import (
        IMAGE_CONTENT_TYPES, _split_host_port, allowed_hostnames, safe_join,
    )
except ImportError:  # script import (python src/tools/serve_review.py)
    from serve_annotator import (  # type: ignore[no-redef]
        IMAGE_CONTENT_TYPES, _split_host_port, allowed_hostnames, safe_join,
    )

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REVIEW_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "court_review.html")
IMAGES_DIR = os.path.join(ROOT, "data", "images")
RESULTS_FILE = os.path.join(ROOT, "data", "inference_results.json")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001


@dataclass
class ReviewConfig:
    images_dir: str = IMAGES_DIR
    results_file: str = RESULTS_FILE
    review_html: str = REVIEW_HTML


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    config: ReviewConfig
    allowed_hosts: set


def make_server(config: ReviewConfig | None = None, host: str = DEFAULT_HOST,
                port: int = DEFAULT_PORT, extra_allowed_hosts=()) -> ThreadedHTTPServer:
    """Create (but do not start) the review HTTP server. ``port=0`` picks a free port."""
    server = ThreadedHTTPServer((host, port), ReviewHandler)
    server.config = config or ReviewConfig()
    server.allowed_hosts = allowed_hostnames(host, extra_allowed_hosts)
    return server


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "CVNReview/1.0"
    sys_version = ""

    def _host_ok(self) -> bool:
        host_hdr = self.headers.get("Host")
        if not host_hdr:
            return False
        host, port = _split_host_port(host_hdr)
        if host not in self.server.allowed_hosts:
            return False
        return (port or 80) == self.server.server_address[1]

    def do_GET(self):
        if not self._host_ok():
            self._json_response({"error": "forbidden host"}, 403)
            return
        cfg = self.server.config
        path = urlsplit(self.path).path

        if path in ("/", "/court_review.html"):
            self._serve_file(cfg.review_html, "text/html; charset=utf-8")
            return

        if path == "/api/results":
            if os.path.isfile(cfg.results_file):
                self._serve_file(cfg.results_file, "application/json; charset=utf-8")
            else:
                self._json_response(
                    {"error": "No results. Run: python src/tools/run_inference.py"}, 404)
            return

        if path.startswith("/images/"):
            fpath = safe_join(cfg.images_dir, unquote(path[len("/images/"):]))
            if fpath is None:
                self._json_response({"error": "forbidden path"}, 403)
                return
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in IMAGE_CONTENT_TYPES:
                self._json_response({"error": "forbidden file type"}, 403)
                return
            self._serve_file(fpath, IMAGE_CONTENT_TYPES[ext])
            return

        self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        self._json_response({"error": "method not allowed"}, 405)

    def _serve_file(self, fpath, content_type):
        if not os.path.isfile(fpath):
            self._json_response({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(os.path.getsize(fpath)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with open(fpath, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        msg = str(args[0]) if args else ""
        if "/images/" not in msg:
            super().log_message(fmt, *args)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Court detection review server")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Interface to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allow-host", action="append", default=[],
                        help="Extra Host header name to accept")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    args = parser.parse_args(argv)

    cfg = ReviewConfig()
    if not os.path.isfile(cfg.results_file):
        print(f"WARNING: No inference results found at {cfg.results_file}")
        print("  Run first: python src/tools/run_inference.py\n")

    server = make_server(cfg, args.host, args.port, args.allow_host)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Court Review running at {url} (bound to {args.host})")
    print(f"  HTML:    {cfg.review_html}")
    print(f"  Images:  {cfg.images_dir}")
    print(f"  Results: {cfg.results_file}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
