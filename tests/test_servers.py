"""Security and behaviour tests for the local annotator / review HTTP servers.

Each server is started in a background thread on an ephemeral loopback port, pointed at
temporary directories, and probed with raw HTTP requests (so paths are sent un-normalized).
"""

import http.client
import json
import threading

import pytest

from src.tools import serve_annotator, serve_review

SECRET = "ROBOFLOW_API_KEY=super-secret"


def _write(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


class Client:
    def __init__(self, server):
        self.host, self.port = server.server_address[:2]

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def get(self, path, headers=None):
        return self.request("GET", path, headers=headers)

    def post_json(self, path, payload, headers=None):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        return self.request("POST", path, body=body, headers=h)


def _run(server):
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True)
    t.start()
    return t


@pytest.fixture
def ann_env(tmp_path):
    root = tmp_path / "repo"
    _write(root / ".env", SECRET)
    _write(root / "README.md", "readme")
    _write(root / "src" / "tools" / "serve_annotator.py", "code")
    tools = root / "src" / "tools"
    _write(tools / "annotator.html", "<html>annotator</html>")
    _write(tools / "dashboard.html", "<html>dashboard</html>")
    data = root / "data"
    _write(data / "images" / "a.jpg", b"\xff\xd8img-a")
    _write(data / "images" / "notes.txt", "not an image")
    (data / "annotations").mkdir(parents=True)
    _write(data / "scraped" / "s1.jpg", b"\xff\xd8scraped-1")
    _write(data / "scraped" / "a.jpg", b"\xff\xd8scraped-a")
    _write(data / "cvn_dataset" / "train" / "images" / "x 1.jpg", b"\xff\xd8train-x")
    _write(data / "cvn_dataset" / "train" / "annotations" / "x 1.json",
           json.dumps({"image_path": "x 1.jpg", "keypoints": [], "visibility": [1, 0]}))
    _write(data / "blender" / "images" / "b.png", b"\x89PNGblender")

    cfg = serve_annotator.AnnotatorConfig(
        images_dir=str(data / "images"),
        annotations_dir=str(data / "annotations"),
        suggestions_dir=str(data / "annotations_suggested"),
        scraped_dir=str(data / "scraped"),
        cvn_dataset_dir=str(data / "cvn_dataset"),
        blender_dir=str(data / "blender"),
        tools_dir=str(tools),
    )
    server = serve_annotator.make_server(cfg, "127.0.0.1", 0)
    _run(server)
    try:
        yield {"root": root, "data": data, "client": Client(server), "server": server}
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def review_env(tmp_path):
    root = tmp_path / "repo"
    _write(root / ".env", SECRET)
    _write(root / "court_review.html", "<html>review</html>")
    _write(root / "data" / "images" / "r.jpg", b"\xff\xd8review")
    _write(root / "data" / "inference_results.json", json.dumps({"results": []}))
    cfg = serve_review.ReviewConfig(
        images_dir=str(root / "data" / "images"),
        results_file=str(root / "data" / "inference_results.json"),
        review_html=str(root / "court_review.html"),
    )
    server = serve_review.make_server(cfg, "127.0.0.1", 0)
    _run(server)
    try:
        yield {"root": root, "client": Client(server), "server": server}
    finally:
        server.shutdown()
        server.server_close()


def _all_files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Annotator: defaults and normal operation
# ---------------------------------------------------------------------------

def test_default_bind_is_loopback():
    server = serve_annotator.make_server(serve_annotator.AnnotatorConfig(), port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
    server = serve_review.make_server(serve_review.ReviewConfig(), port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_normal_save_works(ann_env):
    c, data = ann_env["client"], ann_env["data"]
    payload = {"filename": "images (10).json", "data": {"image_path": "images (10).jpeg", "k": 1}}
    status, body = c.post_json("/api/save", payload)
    assert status == 200, body
    saved = data / "annotations" / "images (10).json"
    assert json.loads(saved.read_text(encoding="utf-8")) == payload["data"]
    # Round-trip through the listing endpoint
    status, body = c.get("/api/annotations")
    assert status == 200
    assert json.loads(body)["images (10).json"]["k"] == 1


def test_suggestions_listing_is_separate_from_annotations(ann_env):
    c, data = ann_env["client"], ann_env["data"]
    status, body = c.get("/api/suggestions")          # folder absent: empty, not an error
    assert status == 200 and json.loads(body) == {}
    _write(data / "annotations_suggested" / "a.json",
           json.dumps({"image_path": "a.jpg", "keypoint_origin": ["suggested"]}))
    status, body = c.get("/api/suggestions")
    assert status == 200 and json.loads(body)["a.json"]["image_path"] == "a.jpg"
    status, body = c.get("/api/annotations")
    assert "a.json" not in json.loads(body)
    status, _ = c.get("/api/suggestions", headers={"Host": "attacker.example"})
    assert status == 403


def test_same_origin_browser_post_allowed(ann_env):
    c = ann_env["client"]
    origin = f"http://localhost:{c.port}"
    status, _ = c.post_json("/api/save", {"filename": "ok.json", "data": {}},
                            headers={"Origin": origin, "Host": f"localhost:{c.port}"})
    assert status == 200


def test_listing_and_image_routes(ann_env):
    c = ann_env["client"]
    status, body = c.get("/api/images")
    assert status == 200 and json.loads(body) == ["a.jpg"]
    assert c.get("/images/a.jpg") == (200, b"\xff\xd8img-a")
    assert c.get("/scraped/s1.jpg")[0] == 200
    assert c.get("/blender/b.png")[0] == 200
    assert c.get("/dataset/train/images/x%201.jpg") == (200, b"\xff\xd8train-x")
    assert c.get("/")[1] == b"<html>annotator</html>"
    assert c.get("/dashboard")[1] == b"<html>dashboard</html>"


def test_dashboard_data_urls_are_quoted(ann_env):
    c = ann_env["client"]
    status, body = c.get("/api/dashboard/data/roboflow-train")
    assert status == 200
    items = json.loads(body)
    assert items[0]["img_url"] == "/dataset/train/images/x%201.jpg"
    # Unknown / traversal source ids are rejected rather than joined into a path
    assert c.get("/api/dashboard/data/roboflow-..%2F..")[0] == 404
    assert c.get("/api/dashboard/data/nope")[0] == 404


# ---------------------------------------------------------------------------
# Annotator: path traversal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", [
    "../x.json",
    "..\\x.json",
    "../../x.json",
    "sub/x.json",
    "%2e%2e%2fx.json",
    "..%2fx.json",
    ".json",
    "..json",
    "x.json\x00.txt",
    "x.txt",
    "",
    None,
    123,
])
def test_save_rejects_bad_filenames(ann_env, fname):
    root = ann_env["root"]
    before = _all_files(root)
    status, _ = ann_env["client"].post_json("/api/save", {"filename": fname, "data": {"p": 1}})
    assert status == 400
    assert _all_files(root) == before


def test_save_rejects_absolute_path(ann_env, tmp_path):
    target = tmp_path / "outside.json"
    status, _ = ann_env["client"].post_json("/api/save", {"filename": str(target), "data": {}})
    assert status == 400
    assert not target.exists()
    status, _ = ann_env["client"].post_json("/api/save", {"filename": "C:\\evil.json", "data": {}})
    assert status == 400


@pytest.mark.parametrize("path", [
    "/images/..%2f..%2f.env",
    "/images/..%2F..%2F..%2F.env",
    "/images/../../.env",
    "/images/..%5c..%5c.env",
    "/images/%2e%2e/%2e%2e/.env",
    "/images/notes.txt",
    "/scraped/..%2fimages%2fa.jpg",
    "/blender/..%2f..%2f..%2f.env",
    "/dataset/..%2f..%2f.env",
    "/dataset/train/images/..%2f..%2f..%2f..%2f.env",
    "/dataset/train/../../../.env",
    "/images/%00a.jpg",
])
def test_get_traversal_rejected(ann_env, path):
    status, body = ann_env["client"].get(path)
    assert status in (403, 404)
    assert SECRET.encode() not in body


@pytest.mark.parametrize("path", [
    "/.env", "/README.md", "/src/tools/serve_annotator.py", "/data/images/a.jpg",
    "/annotator.html/../.env", "/?x=/.env",
])
def test_no_repo_root_fallback(ann_env, path):
    status, body = ann_env["client"].get(path)
    if path.startswith("/?"):
        assert body == b"<html>annotator</html>"  # query string ignored, still the app page
        return
    assert status == 404
    assert SECRET.encode() not in body and b"code" != body and b"readme" != body


@pytest.mark.parametrize("endpoint", ["/api/scraped/keep", "/api/scraped/discard"])
@pytest.mark.parametrize("fname", [
    "../../.env", "..\\..\\.env", "../images/a.jpg", "s1.txt", "/abs/s1.jpg", ".env",
])
def test_scraped_move_delete_traversal(ann_env, endpoint, fname):
    root = ann_env["root"]
    before = _all_files(root)
    status, _ = ann_env["client"].post_json(endpoint, {"filename": fname})
    assert status in (400, 404)
    assert _all_files(root) == before


def test_scraped_keep_does_not_overwrite(ann_env):
    c, data = ann_env["client"], ann_env["data"]
    status, body = c.post_json("/api/scraped/keep", {"filename": "a.jpg"})
    assert status == 200
    assert json.loads(body)["filename"] == "a_1.jpg"
    assert (data / "images" / "a.jpg").read_bytes() == b"\xff\xd8img-a"
    assert (data / "images" / "a_1.jpg").read_bytes() == b"\xff\xd8scraped-a"
    status, _ = c.post_json("/api/scraped/discard", {"filename": "s1.jpg"})
    assert status == 200
    assert not (data / "scraped" / "s1.jpg").exists()


# ---------------------------------------------------------------------------
# Annotator: CSRF / DNS rebinding / body validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headers", [
    {"Origin": "http://evil.example"},
    {"Origin": "null"},
    {"Origin": "http://localhost:1"},
    {"Origin": "https://localhost"},
    {"Sec-Fetch-Site": "cross-site"},
    {"Host": "evil.example"},
])
def test_cross_origin_post_rejected(ann_env, headers):
    data = ann_env["data"]
    status, _ = ann_env["client"].post_json(
        "/api/save", {"filename": "csrf.json", "data": {}}, headers=headers)
    assert status == 403
    assert not (data / "annotations" / "csrf.json").exists()


def test_non_json_content_type_rejected(ann_env):
    c, data = ann_env["client"], ann_env["data"]
    body = json.dumps({"filename": "form.json", "data": {}}).encode("utf-8")
    status, _ = c.request("POST", "/api/save", body=body,
                          headers={"Content-Type": "text/plain"})
    assert status == 415
    assert not (data / "annotations" / "form.json").exists()


def test_rebinding_get_rejected(ann_env):
    status, _ = ann_env["client"].get("/api/annotations", headers={"Host": "attacker.example"})
    assert status == 403


def test_body_validation(ann_env):
    c = ann_env["client"]
    # Declared length over the limit is refused before the body is read.
    too_big = str(serve_annotator.MAX_BODY_BYTES + 1)
    assert c.post_json("/api/save", b"{}", headers={"Content-Length": too_big})[0] == 413
    assert c.post_json("/api/save", b"{not json")[0] == 400
    assert c.post_json("/api/save", b"[1, 2]")[0] == 400
    assert c.post_json("/api/save", {"filename": "d.json", "data": [1]})[0] == 400
    assert c.post_json("/api/nope", {})[0] == 404


# ---------------------------------------------------------------------------
# Review server
# ---------------------------------------------------------------------------

def test_review_normal_routes(review_env):
    c = review_env["client"]
    assert c.get("/")[1] == b"<html>review</html>"
    status, body = c.get("/api/results")
    assert status == 200 and json.loads(body) == {"results": []}
    assert c.get("/images/r.jpg") == (200, b"\xff\xd8review")


@pytest.mark.parametrize("path", [
    "/images/..%2f..%2f.env",
    "/images/../../.env",
    "/images/..%5c..%5c.env",
    "/.env",
    "/court_review.html/../.env",
    "/data/images/r.jpg",
])
def test_review_traversal_and_fallback(review_env, path):
    status, body = review_env["client"].get(path)
    assert status in (403, 404)
    assert SECRET.encode() not in body


def test_review_rejects_foreign_host_and_post(review_env):
    c = review_env["client"]
    assert c.get("/api/results", headers={"Host": "evil.example"})[0] == 403
    assert c.post_json("/api/results", {})[0] == 405
