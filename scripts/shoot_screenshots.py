"""Capture Streamlit UI screenshots for the README (needs the app running on :8501).

    pip install playwright && python -m playwright install chromium
    python scripts/shoot_screenshots.py
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
URL = "http://localhost:8501"


def _tab(page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    time.sleep(3.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1360, "height": 1000})
        page.goto(URL, wait_until="networkidle")
        time.sleep(4)

        _tab(page, "Metrics")
        page.screenshot(path=str(OUT / "metrics.png"), full_page=True)

        _tab(page, "Review Queue")
        page.screenshot(path=str(OUT / "review_queue.png"), full_page=True)

        _tab(page, "Search")
        page.get_by_role("button", name="Search").click()
        time.sleep(5)
        page.screenshot(path=str(OUT / "search.png"), full_page=True)

        _tab(page, "Upload & Process")
        page.screenshot(path=str(OUT / "upload.png"), full_page=True)

        b.close()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.relative_to(OUT.parent.parent)}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
