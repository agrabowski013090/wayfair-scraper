#!/usr/bin/env python3
"""
Wayfair Electric Bike / Electric Scooter compliance scraper.

Crawls the public wayfair.com search results for "electric bike" and
"electric scooter", follows every product link, and extracts:
    - Wayfair SKU
    - Product name
    - Brand
    - URL
    - Classification (actual electric bike / actual electric scooter / excluded)

Design notes
------------
* Uses Playwright (Chromium) because Wayfair's storefront is a client-rendered
  Next.js app - a plain HTTP GET will not contain the product grid.
* This is a NORMAL, front-door crawl: it loads pages the same way a browser
  would, honors robots.txt, and rate-limits itself. It does NOT attempt to
  defeat bot detection, stash data in browser-only state to dodge output
  limits, or otherwise route around technical restrictions. Results are
  written straight to disk as they're found.
* Resumable: progress (which listing pages have been crawled, which product
  URLs have already been visited) is persisted to a checkpoint JSON file
  after every item, so killing the process and re-running the script picks
  up where it left off instead of re-scraping everything.
* Polite by default: sequential (no concurrent tabs), randomized delay
  between every navigation, exponential backoff on errors, and a hard stop
  if too many consecutive failures happen in a row (likely sign of a block -
  better to stop than hammer the site).

Usage
-----
    pip install playwright
    playwright install chromium

    python wayfair_ebike_escooter_scraper.py \
        --keywords "electric bike" "electric scooter" \
        --output ebike_escooter_report.csv \
        --checkpoint ebike_escooter_checkpoint.json \
        --min-delay 2.5 --max-delay 5.0 \
        --headless

Re-running the same command after an interruption (Ctrl-C, crash, network
blip) resumes automatically from the checkpoint file.
"""

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urljoin

from playwright.sync_api import (
    Page,
    TimeoutError as PWTimeoutError,
    sync_playwright,
)

BASE_URL = "https://www.wayfair.com"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

SELECTORS = {
    "results_container": '[data-enzyme-id="BrowseGrid"], #BrowseGrid, main',
    "product_link": 'a[href*=".html"]',
    "next_page": 'a[aria-label="Next"], a[rel="next"]',
    "jsonld": 'script[type="application/ld+json"]',
    "title_h1": "h1",
    "og_title": 'meta[property="og:title"]',
    "breadcrumbs": '[data-enzyme-id="Breadcrumbs"] a, nav[aria-label="breadcrumb"] a',
}

SKU_URL_RE = re.compile(r"-([A-Z]{1,6}\d{2,8}(?:-[A-Z0-9]+)?)\.html", re.IGNORECASE)
SKU_TEXT_RE = re.compile(r"\bSKU[:\s]*([A-Za-z0-9\-]{4,20})\b", re.IGNORECASE)

SPEC_TERM_RE = re.compile(
    r"\b(\d+\s?(mph|w|watt|wh|volt|v)\b|battery|throttle|pedal[\s-]?assist|"
    r"lithium|charge time|range\s?(per\s?charge|of)?\s?\d|class\s?[123]\b|"
    r"motor)",
    re.IGNORECASE,
)

_ELECTRIC_RE = re.compile(r"\belectric\b", re.IGNORECASE)
_EBIKE_TOKEN_RE = re.compile(r"\be-?bike\b", re.IGNORECASE)
_ESCOOTER_TOKEN_RE = re.compile(r"\be-?scooter\b", re.IGNORECASE)
_BIKE_WORD_RE = re.compile(r"\b(bike|bicycle|trike|cycle)\b", re.IGNORECASE)
_SCOOTER_WORD_RE = re.compile(r"\bscooter\b", re.IGNORECASE)


def _is_electric_bike(text: str) -> bool:
    return bool(_EBIKE_TOKEN_RE.search(text)) or (
        bool(_ELECTRIC_RE.search(text)) and bool(_BIKE_WORD_RE.search(text))
    )


def _is_electric_scooter(text: str) -> bool:
    return bool(_ESCOOTER_TOKEN_RE.search(text)) or (
        bool(_ELECTRIC_RE.search(text)) and bool(_SCOOTER_WORD_RE.search(text))
    )

EXCLUSION_RE = re.compile(
    r"\b(exercise|stationary|spin\s?bike|recumbent|indoor\s?cycling|"
    r"accessor(y|ies)|helmet|rack|storage|shed|cover\b|lock\b|basket|"
    r"pump\b|repair\s?stand|training\s?wheel|replacement\s?(battery|tire|"
    r"part)|spare\s?part|decor|wall\s?art|sign\b|pillow|figurine|"
    r"\btoy\b|ride-?on\s?(car|toy)|kids?\s?ride-?on|costume|mount\b|"
    r"trailer\b|stroller\b|charger\s?only|carrying\s?case|travel\s?bag)",
    re.IGNORECASE,
)


def classify(name: str, breadcrumbs: str, description: str) -> str:
    text = " ".join([name or "", breadcrumbs or "", description or ""])
    if EXCLUSION_RE.search(text):
        return "excluded"

    has_spec = bool(SPEC_TERM_RE.search(text))
    is_bike = _is_electric_bike(text)
    is_scooter = _is_electric_scooter(text)

    if is_bike and has_spec:
        return "electric_bike"
    if is_scooter and has_spec:
        return "electric_scooter"
    if is_bike or is_scooter:
        return "needs_review"
    return "excluded"


@dataclass
class Checkpoint:
    path: Path
    visited_product_urls: set = field(default_factory=set)
    completed_listing_pages: dict = field(default_factory=dict)
    finished_keywords: set = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        if path.exists():
            data = json.loads(path.read_text())
            return cls(
                path=path,
                visited_product_urls=set(data.get("visited_product_urls", [])),
                completed_listing_pages=data.get("completed_listing_pages", {}),
                finished_keywords=set(data.get("finished_keywords", [])),
            )
        return cls(path=path)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "visited_product_urls": sorted(self.visited_product_urls),
                    "completed_listing_pages": self.completed_listing_pages,
                    "finished_keywords": sorted(self.finished_keywords),
                },
                indent=2,
            )
        )
        tmp.replace(self.path)


class RateLimiter:
    def __init__(self, min_delay: float, max_delay: float):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.consecutive_failures = 0

    def wait(self):
        time.sleep(random.uniform(self.min_delay, self.max_delay))

    def note_success(self):
        self.consecutive_failures = 0

    def note_failure(self):
        self.consecutive_failures += 1
        backoff = min(60, self.min_delay * (2 ** self.consecutive_failures))
        time.sleep(backoff)


def check_robots_allowed(url: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(ROBOTS_URL)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def goto_with_retry(page: Page, url: str, limiter: RateLimiter, retries: int = 3):
    for attempt in range(1, retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            limiter.note_success()
            return True
        except PWTimeoutError:
            limiter.note_failure()
            if attempt == retries:
                return False
    return False


def extract_product_links(page: Page) -> list:
    try:
        page.wait_for_selector(SELECTORS["product_link"], timeout=8000)
    except PWTimeoutError:
        return []
    hrefs = page.eval_on_selector_all(
        SELECTORS["product_link"], "els => els.map(e => e.href)"
    )
    seen = []
    for h in hrefs:
        if h and h.endswith(".html") and "/keyword.php" not in h and h not in seen:
            seen.append(h)
    return seen


def extract_jsonld_product(page: Page) -> Optional[dict]:
    try:
        scripts = page.eval_on_selector_all(
            SELECTORS["jsonld"], "els => els.map(e => e.textContent)"
        )
    except Exception:
        return None
    for raw in scripts:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for c in candidates:
            if not isinstance(c, dict):
                continue
            type_field = c.get("@type")
            is_product = type_field == "Product" or (
                isinstance(type_field, list) and "Product" in type_field
            )
            if is_product:
                return c
    return None


def extract_product_details(page: Page, url: str, debug_dir: Optional[Path]) -> dict:
    jsonld = extract_jsonld_product(page)

    name = None
    brand = None
    sku = None

    if jsonld:
        name = jsonld.get("name")
        brand_field = jsonld.get("brand")
        if isinstance(brand_field, dict):
            brand = brand_field.get("name")
        elif isinstance(brand_field, str):
            brand = brand_field
        sku = jsonld.get("sku") or jsonld.get("mpn")

    if not name:
        try:
            name = page.eval_on_selector(
                SELECTORS["og_title"], "el => el.content"
            )
        except Exception:
            pass
    if not name:
        try:
            name = page.eval_on_selector(SELECTORS["title_h1"], "el => el.textContent.trim()")
        except Exception:
            name = None

    if not sku:
        m = SKU_URL_RE.search(url)
        if m:
            sku = m.group(1)
    if not sku:
        try:
            body_text = page.eval_on_selector("body", "el => el.innerText")
            m = SKU_TEXT_RE.search(body_text or "")
            if m:
                sku = m.group(1)
        except Exception:
            pass

    breadcrumbs = ""
    try:
        crumbs = page.eval_on_selector_all(
            SELECTORS["breadcrumbs"], "els => els.map(e => e.textContent.trim())"
        )
        breadcrumbs = " > ".join(crumbs)
    except Exception:
        pass

    description = jsonld.get("description", "") if jsonld else ""

    if debug_dir and not (name and sku):
        debug_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", url)[-100:]
        (debug_dir / f"{safe_name}.html").write_text(page.content(), encoding="utf-8")

    return {
        "url": url,
        "sku": sku or "",
        "name": name or "",
        "brand": brand or "",
        "breadcrumbs": breadcrumbs,
        "description": description or "",
    }


def crawl(args):
    output_path = Path(args.output)
    checkpoint = Checkpoint.load(Path(args.checkpoint))
    limiter = RateLimiter(args.min_delay, args.max_delay)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    csv_file = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["sku", "name", "brand", "url", "classification", "breadcrumbs", "keyword"],
    )
    if write_header:
        writer.writeheader()
        csv_file.flush()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()

        for keyword in args.keywords:
            if keyword in checkpoint.finished_keywords:
                print(f"[skip] '{keyword}' already fully crawled per checkpoint")
                continue

            print(f"=== Crawling keyword: {keyword!r} ===")
            encoded_kw = quote(keyword)
            start_page = checkpoint.completed_listing_pages.get(keyword, 0) + 1
            page_num = start_page
            consecutive_empty = 0

            while True:
                if args.max_pages and page_num > args.max_pages:
                    print(f"[stop] hit --max-pages={args.max_pages} for {keyword!r}")
                    break

                listing_url = f"{BASE_URL}/keyword.php?keyword={encoded_kw}&curpage={page_num}"
                if not check_robots_allowed(listing_url):
                    print(f"[robots] disallowed, stopping: {listing_url}")
                    break

                ok = goto_with_retry(page, listing_url, limiter)
                limiter.wait()
                if not ok:
                    print(f"[error] could not load page {page_num}, stopping keyword")
                    break

                links = extract_product_links(page)
                new_links = [l for l in links if l not in checkpoint.visited_product_urls]

                if not links:
                    consecutive_empty += 1
                    print(f"[page {page_num}] no product links found ({consecutive_empty} empty in a row)")
                    if consecutive_empty >= 2:
                        print(f"[end] reached end of results for {keyword!r} at page {page_num}")
                        break
                    page_num += 1
                    continue
                consecutive_empty = 0

                print(f"[page {page_num}] {len(links)} product links, {len(new_links)} new")

                for product_url in new_links:
                    if not check_robots_allowed(product_url):
                        continue
                    ok = goto_with_retry(page, product_url, limiter)
                    limiter.wait()
                    if not ok:
                        continue
                    details = extract_product_details(page, product_url, debug_dir)
                    label = classify(
                        details["name"], details["breadcrumbs"], details["description"]
                    )
                    writer.writerow(
                        {
                            "sku": details["sku"],
                            "name": details["name"],
                            "brand": details["brand"],
                            "url": details["url"],
                            "classification": label,
                            "breadcrumbs": details["breadcrumbs"],
                            "keyword": keyword,
                        }
                    )
                    csv_file.flush()
                    checkpoint.visited_product_urls.add(product_url)
                    checkpoint.save()

                checkpoint.completed_listing_pages[keyword] = page_num
                checkpoint.save()
                page_num += 1

            checkpoint.finished_keywords.add(keyword)
            checkpoint.save()

        browser.close()

    csv_file.close()
    print(f"Done. Results written to {output_path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keywords", nargs="+", default=["electric bike", "electric scooter"])
    ap.add_argument("--output", default="ebike_escooter_report.csv")
    ap.add_argument("--checkpoint", default="ebike_escooter_checkpoint.json")
    ap.add_argument("--min-delay", type=float, default=2.5, help="Minimum delay in seconds between navigations")
    ap.add_argument("--max-delay", type=float, default=5.0, help="Maximum delay in seconds between navigations")
    ap.add_argument("--max-pages", type=int, default=None, help="Max listing pages per keyword (testing)")
    ap.add_argument("--headless", action="store_true", help="Run Chromium in headless mode")
    ap.add_argument("--debug-dir", default=None, help="Dump HTML for products missing name/sku")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        crawl(args)
    except KeyboardInterrupt:
        print("\n[interrupted] progress saved to checkpoint; re-run to resume.", file=sys.stderr)
        sys.exit(130)
