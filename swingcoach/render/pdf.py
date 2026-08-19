"""PDF export via headless Chromium (Playwright), for sharing wherever HTML
isn't accepted -- LinkedIn attachments, email clients that choke on it, etc.

Renders whatever HTML it's pointed at; report.py's PDF path specifically
generates a video-free report first (generate_report(..., include_video=False))
because a PDF can't play the embedded <video> elements -- they render as a
dead black box, confirmed directly by rendering a real report to PDF and
looking at the output. The report already tells the same story with three
key-frame stills per swing, so nothing is actually lost by leaving video out
of this path specifically; the HTML report keeps video for anyone opening it
in a browser.
"""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def _find_chromium() -> str:
    """Playwright's own browser auto-discovery expects a specific pinned
    revision that doesn't match what's pre-installed in this environment
    (confirmed: default launch() fails looking for revision 1234, but only
    1194 exists on disk) -- so this searches PLAYWRIGHT_BROWSERS_PATH
    directly instead of relying on playwright's version-pinned lookup, and
    never runs `playwright install` (which would try to download a browser
    this environment already provides)."""
    browsers_path = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"))
    matches = sorted(browsers_path.glob("chromium-*/chrome-linux/chrome"))
    if not matches:
        raise RuntimeError(
            f"no chromium executable found under {browsers_path} "
            "(expected chromium-<revision>/chrome-linux/chrome)"
        )
    return str(matches[-1])


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_find_chromium())
        try:
            page = browser.new_page()
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_timeout(500)  # let matplotlib/base64 images settle
            page.pdf(path=str(pdf_path), print_background=True)
        finally:
            browser.close()
