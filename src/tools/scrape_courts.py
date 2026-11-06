"""Scrape badminton court images from Bing/Google for training data.

Usage:  python -m src.tools.scrape_courts [--max 200] [--engine bing] [--output data/scraped]

Every downloaded image gets a provenance record (source URL, query, engine, UTC timestamp,
SHA-256) appended to ``<output>/provenance.jsonl``. Images whose content hash already exists in
the output directory are skipped, and existing files are never overwritten (new files are
numbered after the highest existing index).
"""

import argparse
import hashlib
import json
import os
import threading
from datetime import datetime, timezone

from icrawler import ImageDownloader
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

QUERIES = [
    "badminton court floor lines",
    "badminton court floor markings",
    "badminton court empty gymnasium",
    "badminton court wooden floor",
    "badminton court synthetic floor",
    "badminton court mat green",
    "badminton match court view wide angle",
    "badminton game playing on court",
    "badminton tournament court aerial",
    "badminton hall multiple courts",
    "badminton court sports hall",
    "indoor badminton court photo",
    "outdoor badminton court concrete",
    "badminton court net full view",
    "BWF badminton court match",
    "community center badminton court",
]

ENGINES = {
    "bing": BingImageCrawler,
    "google": GoogleImageCrawler,
}

PROVENANCE_FILE = "provenance.jsonl"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def existing_hashes(output_dir: str) -> set[str]:
    """Content hashes of the images already present in ``output_dir``."""
    hashes = set()
    if not os.path.isdir(output_dir):
        return hashes
    for fname in os.listdir(output_dir):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        with open(os.path.join(output_dir, fname), "rb") as f:
            hashes.add(sha256_bytes(f.read()))
    return hashes


class ProvenanceLog:
    """Thread-safe content-hash dedupe + JSONL provenance writer shared by downloader threads."""

    def __init__(self, output_dir: str, engine: str):
        self.path = os.path.join(output_dir, PROVENANCE_FILE)
        self.engine = engine
        self.query = ""
        self.seen = existing_hashes(output_dir)
        self.duplicates = 0
        self._lock = threading.Lock()

    def claim(self, digest: str) -> bool:
        """Return True if ``digest`` is new (and reserve it), False if it is a duplicate."""
        with self._lock:
            if digest in self.seen:
                self.duplicates += 1
                return False
            self.seen.add(digest)
            return True

    def record(self, filename: str, source_url: str, digest: str, img_size=None):
        rec = {
            "filename": filename,
            "source_url": source_url,
            "query": self.query,
            "engine": self.engine,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": digest,
        }
        if img_size is not None:
            rec["image_size_wh"] = list(img_size)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class ProvenanceDownloader(ImageDownloader):
    """ImageDownloader that skips duplicate content and logs where each file came from."""

    provenance: ProvenanceLog | None = None

    def keep_file(self, task, response, **kwargs):
        if not super().keep_file(task, response, **kwargs):
            return False
        if self.provenance is None:
            return True
        digest = sha256_bytes(response.content)
        if not self.provenance.claim(digest):
            self.logger.info("skip duplicate content %s", task.get("file_url"))
            return False
        task["sha256"] = digest
        return True

    def process_meta(self, task):
        if self.provenance is None or not task.get("success") or not task.get("filename"):
            return
        self.provenance.record(
            filename=task["filename"],
            source_url=task.get("file_url", ""),
            digest=task.get("sha256", ""),
            img_size=task.get("img_size"),
        )


def _count_images(output_dir: str) -> int:
    return sum(1 for f in os.listdir(output_dir)
               if os.path.splitext(f)[1].lower() in IMAGE_EXTS)


def scrape(output_dir: str, max_per_query: int, engine: str):
    os.makedirs(output_dir, exist_ok=True)

    crawler_cls = ENGINES[engine]
    provenance = ProvenanceLog(output_dir, engine)
    start_count = _count_images(output_dir)

    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        before = _count_images(output_dir)
        provenance.query = query
        crawler = crawler_cls(downloader_cls=ProvenanceDownloader,
                              storage={"root_dir": output_dir})
        crawler.downloader.provenance = provenance
        crawler.crawl(
            keyword=query,
            max_num=max_per_query,
            min_size=(200, 200),
            file_idx_offset="auto",  # continue after the highest existing index; never overwrite
        )

        new_files = _count_images(output_dir) - before
        print(f"  Downloaded {new_files} images "
              f"(total new: {_count_images(output_dir) - start_count})")

    total = _count_images(output_dir) - start_count
    print(f"\nDone! {total} new images saved to {output_dir} "
          f"({provenance.duplicates} duplicates skipped)")
    print(f"Provenance: {provenance.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape badminton court images")
    parser.add_argument("--output", default="data/scraped", help="Output directory")
    parser.add_argument("--max", type=int, default=50, help="Max images per query")
    parser.add_argument("--engine", choices=["bing", "google"], default="bing",
                        help="Search engine (bing recommended, google blocks aggressively)")
    args = parser.parse_args()

    scrape(args.output, args.max, args.engine)
