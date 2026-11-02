"""Local server for the CourtVisionNet annotator.

Run:  python src/tools/serve_annotator.py [--host 127.0.0.1] [--port 8000] [--no-browser]
Open: http://127.0.0.1:8000

Security model: this is a single-user local tool. By default it binds to the loopback interface
only, serves an explicit allowlist of routes (never the repository root), confines every file
access to its configured data directory, and rejects cross-origin / DNS-rebinding requests.
"""

import argparse
import json
import os
import re
import shutil
import tempfile
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import quote, unquote, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(ROOT, "data", "images")
ANNOTATIONS_DIR = os.path.join(ROOT, "data", "annotations")
# Machine pre-annotations (keypoint_origin "suggested"), shown for review when
# an image has no annotation yet. Saving writes to ANNOTATIONS_DIR as usual.
SUGGESTIONS_DIR = os.path.join(ROOT, "data", "annotations_suggested")
SCRAPED_DIR = os.path.join(ROOT, "data", "scraped")
CVN_DATASET_DIR = os.path.join(ROOT, "data", "cvn_dataset")
BLENDER_DIR = os.path.join(ROOT, "data", "blender")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 2 * 1024 * 1024  # annotation JSON is a few KB; 2 MB is generous

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".bmp": "image/bmp", ".webp": "image/webp",
}
DATASET_SPLITS = ("train", "valid", "test")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Annotation filenames: a single path component made of word chars and a few safe symbols.
ANNOTATION_NAME_RE = re.compile(r"^[\w\-. ()+,@\[\]]+\.json$")
# Image filenames (scraped keep/discard): same alphabet, image extension checked separately.
IMAGE_NAME_RE = re.compile(r"^[\w\-. ()+,@\[\]]+$")


# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------

def _is_within(base: str, candidate: str) -> bool:
    base_real = os.path.normcase(os.path.realpath(base))
    cand_real = os.path.normcase(os.path.realpath(candidate))
    try:
        return os.path.commonpath([base_real, cand_real]) == base_real
    except ValueError:  # different drives on Windows
        return False


def safe_join(base: str, relpath: str, allow_subdirs: bool = False) -> str | None:
    """Join an already-URL-decoded relative path onto ``base``; None if it escapes ``base``.

    With ``allow_subdirs=False`` only a single path component (a plain filename) is accepted.
    """
    if not relpath or "\x00" in relpath:
        return None
    rel = relpath.replace("\\", "/")
    if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        return None
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    if not allow_subdirs and len(parts) != 1:
        return None
    candidate = os.path.join(base, *parts)
    if not _is_within(base, candidate):
        return None
    return candidate


def safe_filename(name, pattern: re.Pattern) -> str | None:
    """Validate a client-supplied filename: one component matching ``pattern``, no leading dot."""
    if not isinstance(name, str) or not name or len(name) > 255:
        return None
    if name.startswith(".") or os.path.basename(name) != name or "/" in name or "\\" in name:
        return None
    if not pattern.match(name):
        return None
    return name


def _unique_path(directory: str, fname: str) -> str:
    """Return a path in ``directory`` for ``fname`` that does not overwrite an existing file."""
    stem, ext = os.path.splitext(fname)
    path = os.path.join(directory, fname)
    i = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{i}{ext}")
        i += 1
    return path


def _list_images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        and os.path.isfile(os.path.join(directory, f))
    )


def _count_json(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(".json")])


# ---------------------------------------------------------------------------
# Configuration / server
# ---------------------------------------------------------------------------

@dataclass
class AnnotatorConfig:
    images_dir: str = IMAGES_DIR
    annotations_dir: str = ANNOTATIONS_DIR
    suggestions_dir: str = SUGGESTIONS_DIR
    scraped_dir: str = SCRAPED_DIR
    cvn_dataset_dir: str = CVN_DATASET_DIR
    blender_dir: str = BLENDER_DIR
    tools_dir: str = TOOLS_DIR


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    config: AnnotatorConfig
    allowed_hosts: set


def allowed_hostnames(host: str, extra=()) -> set:
    names = set(LOOPBACK_HOSTS)
    if host and host not in ("0.0.0.0", "::", ""):
        names.add(host.lower())
    names.update(h.lower() for h in extra)
    return names


def make_server(config: AnnotatorConfig | None = None, host: str = DEFAULT_HOST,
                port: int = DEFAULT_PORT, extra_allowed_hosts=()) -> ThreadedHTTPServer:
    """Create (but do not start) the annotator HTTP server. ``port=0`` picks a free port."""
    server = ThreadedHTTPServer((host, port), AnnotatorHandler)
    server.config = config or AnnotatorConfig()
    server.allowed_hosts = allowed_hostnames(host, extra_allowed_hosts)
    return server


def _split_host_port(value: str):
    """Parse 'host[:port]' (IPv6 in brackets allowed) -> (lowercase host, port or None)."""
    try:
        parts = urlsplit("//" + value)
        return (parts.hostname or "").lower(), parts.port
    except ValueError:
        return None, None


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class AnnotatorHandler(BaseHTTPRequestHandler):
    server_version = "CVNAnnotator/1.0"
    sys_version = ""

    @property
    def cfg(self) -> AnnotatorConfig:
        return self.server.config

    # -- request validation -------------------------------------------------

    def _host_ok(self) -> bool:
        """Reject requests whose Host header is not ours (DNS-rebinding defence)."""
        host_hdr = self.headers.get("Host")
        if not host_hdr:
            return False
        host, port = _split_host_port(host_hdr)
        if host not in self.server.allowed_hosts:
            return False
        return (port or 80) == self.server.server_address[1]

    def _origin_ok(self) -> bool:
        """For state-changing requests: Origin (if sent) must be this server."""
        if self.headers.get("Sec-Fetch-Site", "") == "cross-site":
            return False
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # non-browser client (curl, tests); Host check still applies
        try:
            parts = urlsplit(origin)
            port = parts.port
        except ValueError:
            return False
        if parts.scheme != "http" or (parts.hostname or "").lower() not in self.server.allowed_hosts:
            return False
        return (port or 80) == self.server.server_address[1]

    def _route_path(self) -> str:
        return urlsplit(self.path).path

    # -- GET ----------------------------------------------------------------

    def do_GET(self):
        if not self._host_ok():
            self._error(403, "forbidden host")
            return
        path = self._route_path()
        cfg = self.cfg

        if path in ("/", "/annotator.html"):
            self._serve_file(os.path.join(cfg.tools_dir, "annotator.html"),
                             "text/html; charset=utf-8")
            return
        if path in ("/dashboard", "/dashboard.html"):
            self._serve_file(os.path.join(cfg.tools_dir, "dashboard.html"),
                             "text/html; charset=utf-8")
            return

        if path == "/api/images":
            self._json_response(_list_images(cfg.images_dir))
            return
        if path == "/api/annotations":
            self._json_response(self._load_annotation_dict(cfg.annotations_dir))
            return
        if path == "/api/suggestions":
            self._json_response(self._load_annotation_dict(cfg.suggestions_dir))
            return
        if path == "/api/scraped":
            files = _list_images(cfg.scraped_dir)
            self._json_response({"files": files, "total": len(files)})
            return
        if path == "/api/dashboard/sources":
            self._json_response(self._get_dashboard_sources())
            return
        if path.startswith("/api/dashboard/data/"):
            source = unquote(path[len("/api/dashboard/data/"):])
            data = self._get_source_data(source)
            if data is None:
                self._error(404, "unknown source")
            else:
                self._json_response(data)
            return

        image_routes = (
            ("/images/", cfg.images_dir, False),
            ("/scraped/", cfg.scraped_dir, False),
            ("/blender/", os.path.join(cfg.blender_dir, "images"), False),
            ("/dataset/", cfg.cvn_dataset_dir, True),
        )
        for prefix, base, subdirs in image_routes:
            if path.startswith(prefix):
                rel = unquote(path[len(prefix):])
                fpath = safe_join(base, rel, allow_subdirs=subdirs)
                if fpath is None:
                    self._error(403, "forbidden path")
                    return
                self._serve_image(fpath)
                return

        self._error(404, "not found")

    def do_HEAD(self):
        self._error(405, "method not allowed")

    # -- POST ---------------------------------------------------------------

    def do_POST(self):
        if not self._host_ok() or not self._origin_ok():
            self._error(403, "cross-origin request rejected")
            return
        path = self._route_path()
        handlers = {
            "/api/save": self._post_save,
            "/api/scraped/keep": self._post_scraped_keep,
            "/api/scraped/discard": self._post_scraped_discard,
        }
        handler = handlers.get(path)
        if handler is None:
            self._error(404, "not found")
            return
        body = self._read_json_body()
        if body is None:
            return  # error already sent
        handler(body)

    def _read_json_body(self):
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._error(415, "Content-Type must be application/json")
            return None
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "Content-Length required")
            return None
        if length < 0 or length > MAX_BODY_BYTES:
            self._error(413, "request body too large")
            return None
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "invalid JSON")
            return None
        if not isinstance(body, dict):
            self._error(400, "JSON body must be an object")
            return None
        return body

    def _post_save(self, body):
        cfg = self.cfg
        fname = safe_filename(body.get("filename"), ANNOTATION_NAME_RE)
        data = body.get("data", {})
        if fname is None:
            self._error(400, "invalid filename")
            return
        if not isinstance(data, dict):
            self._error(400, "data must be a JSON object")
            return
        os.makedirs(cfg.annotations_dir, exist_ok=True)
        fpath = safe_join(cfg.annotations_dir, fname)
        if fpath is None:
            self._error(400, "invalid filename")
            return
        # Atomic write: temp file in the same directory, then replace.
        fd, tmp = tempfile.mkstemp(dir=cfg.annotations_dir, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, fpath)
        except OSError as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            self._error(500, f"write failed: {e.strerror}")
            return
        self._json_response({"ok": True, "filename": fname})

    def _scraped_source(self, body):
        """Validate the filename of a scraped image; returns its path or sends an error."""
        fname = safe_filename(body.get("filename"), IMAGE_NAME_RE)
        if fname is None or os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            self._error(400, "invalid filename")
            return None
        src = safe_join(self.cfg.scraped_dir, fname)
        if src is None:
            self._error(400, "invalid filename")
            return None
        if not os.path.isfile(src):
            self._error(404, "file not found")
            return None
        return src

    def _post_scraped_keep(self, body):
        src = self._scraped_source(body)
        if src is None:
            return
        os.makedirs(self.cfg.images_dir, exist_ok=True)
        dst = _unique_path(self.cfg.images_dir, os.path.basename(src))
        if not _is_within(self.cfg.images_dir, dst):
            self._error(400, "invalid filename")
            return
        shutil.move(src, dst)
        self._json_response({"ok": True, "filename": os.path.basename(dst)})

    def _post_scraped_discard(self, body):
        src = self._scraped_source(body)
        if src is None:
            return
        os.remove(src)
        self._json_response({"ok": True})

    # -- data helpers ---------------------------------------------------------

    @staticmethod
    def _load_annotation_dict(ann_dir):
        anns = {}
        if not os.path.isdir(ann_dir):
            return anns
        for f in os.listdir(ann_dir):
            if not f.endswith(".json") or f.startswith("."):
                continue
            try:
                with open(os.path.join(ann_dir, f), encoding="utf-8") as fh:
                    anns[f] = json.load(fh)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        return anns

    def _get_dashboard_sources(self):
        cfg = self.cfg
        sources = []
        n = _count_json(cfg.annotations_dir)
        if n:
            sources.append({"id": "mine", "name": "My Annotations", "count": n})
        for split in DATASET_SPLITS:
            n = _count_json(os.path.join(cfg.cvn_dataset_dir, split, "annotations"))
            if n:
                sources.append({"id": f"roboflow-{split}", "name": f"Roboflow {split}", "count": n})
        n = _count_json(os.path.join(cfg.blender_dir, "annotations"))
        if n:
            sources.append({"id": "blender", "name": "Blender Synthetic", "count": n})
        return sources

    def _get_source_data(self, source):
        cfg = self.cfg
        if source == "mine":
            return self._load_cvn_annotations(cfg.annotations_dir, "/images/")
        if source.startswith("roboflow-"):
            split = source[len("roboflow-"):]
            if split not in DATASET_SPLITS:
                return None
            ann_dir = os.path.join(cfg.cvn_dataset_dir, split, "annotations")
            return self._load_cvn_annotations(ann_dir, f"/dataset/{split}/images/")
        if source == "blender":
            return self._load_cvn_annotations(os.path.join(cfg.blender_dir, "annotations"),
                                              "/blender/")
        return None

    @staticmethod
    def _load_cvn_annotations(ann_dir, img_prefix):
        items = []
        if not os.path.isdir(ann_dir):
            return items
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(ann_dir, fname), encoding="utf-8") as f:
                    ann = json.load(f)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(ann, dict):
                continue
            img_name = os.path.basename(str(ann.get("image_path", fname[:-5] + ".jpg")))
            vis = ann.get("visibility", [])
            n_vis = sum(1 for v in vis if v)
            items.append({
                "filename": img_name,
                "img_url": img_prefix + quote(img_name),
                "keypoints": ann.get("keypoints", []),
                "visibility": vis,
                "n_visible": n_vis,
            })
        return items

    # -- response helpers ---------------------------------------------------

    def _send_common_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")

    def _serve_image(self, fpath):
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in IMAGE_CONTENT_TYPES:
            self._error(403, "forbidden file type")
            return
        self._serve_file(fpath, IMAGE_CONTENT_TYPES[ext])

    def _serve_file(self, fpath, content_type):
        if not os.path.isfile(fpath):
            self._error(404, "not found")
            return
        size = os.path.getsize(fpath)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self._send_common_headers()
        self.end_headers()
        with open(fpath, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_common_headers()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message):
        self._json_response({"error": message}, code)

    def log_message(self, format, *args):
        msg = str(args[0]) if args else ""
        if not any(p in msg for p in ("/images/", "/scraped/", "/dataset/", "/blender/")):
            super().log_message(format, *args)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CourtVisionNet browser annotator server")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Interface to bind (default 127.0.0.1; use 0.0.0.0 only on a "
                             "trusted network)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allow-host", action="append", default=[],
                        help="Extra Host header name to accept (e.g. this machine's LAN IP)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    args = parser.parse_args(argv)

    cfg = AnnotatorConfig()
    for d in (cfg.images_dir, cfg.annotations_dir, cfg.scraped_dir):
        os.makedirs(d, exist_ok=True)

    server = make_server(cfg, args.host, args.port, args.allow_host)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    print(f"Annotator running at {url} (bound to {args.host})")
    print(f"  Images:      {cfg.images_dir}")
    print(f"  Annotations: {cfg.annotations_dir}")
    if os.path.isdir(cfg.suggestions_dir):
        print(f"  Suggestions: {cfg.suggestions_dir} (shown for review on unannotated images)")
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
