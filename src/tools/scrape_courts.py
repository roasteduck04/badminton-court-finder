"""Scrape badminton court images from Bing/Google for training data.

Usage:  python -m src.tools.scrape_courts [--max 200] [--engine bing]
"""

import argparse
import os

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


def scrape(output_dir: str, max_per_query: int, engine: str):
    os.makedirs(output_dir, exist_ok=True)

    crawler_cls = ENGINES[engine]
    total = 0

    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        crawler = crawler_cls(storage={"root_dir": output_dir})
        crawler.crawl(
            keyword=query,
            max_num=max_per_query,
            min_size=(200, 200),
            file_idx_offset=total,
        )

        new_files = len(os.listdir(output_dir)) - total
        total += new_files
        print(f"  Downloaded {new_files} images (total: {total})")

    print(f"\nDone! {total} images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape badminton court images")
    parser.add_argument("--output", default="data/scraped", help="Output directory")
    parser.add_argument("--max", type=int, default=50, help="Max images per query")
    parser.add_argument("--engine", choices=["bing", "google"], default="bing",
                        help="Search engine (bing recommended, google blocks aggressively)")
    args = parser.parse_args()

    scrape(args.output, args.max, args.engine)
