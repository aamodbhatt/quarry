"""Record the README visuals from the real web UI running the offline scripted demo.

Starts the FastAPI app in-process on a free port (scripted model, demo warehouse), drives
it with Playwright through the locally installed Microsoft Edge, and writes:

* ``docs/assets/hero.png``      the first demo question, answered (repair loop visible)
* ``docs/assets/follow-up.png`` the follow-up question, resolved through memory
* ``docs/assets/guard.png``     a delete request refused by the SQL guard
* ``docs/assets/demo.gif``      the whole sequence as a slideshow

Usage: ``uv run --group docs python scripts/capture_ui.py`` (needs data/warehouse.sqlite).
"""

from __future__ import annotations

import socket
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from PIL import Image
from playwright.sync_api import Page, ViewportSize, sync_playwright

from quarry.agent import open_agent
from quarry.config import QuarryConfig
from quarry.demo import DEMO_QUESTIONS
from quarry.web import create_app

ASSETS = Path("docs/assets")
VIEWPORT: ViewportSize = {"width": 1440, "height": 860}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ask(page: Page, question: str, frames: list[Image.Image], shots: Path) -> None:
    box = page.locator("#question")
    box.click()
    box.type(question, delay=18)
    frames.append(_grab(page, shots, f"typed-{len(frames)}"))
    count = page.locator(".msg.bot").count()
    page.keyboard.press("Enter")
    page.wait_for_function(
        f"document.querySelectorAll('.msg.bot').length > {count}", timeout=30_000
    )
    page.wait_for_timeout(1_800)  # let the chart and trace waterfall render


def _grab(page: Page, shots: Path, name: str) -> Image.Image:
    path = shots / f"{name}.png"
    page.screenshot(path=str(path))
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    database = Path("data/warehouse.sqlite")
    if not database.exists():
        raise SystemExit("data/warehouse.sqlite is missing; run `uv run quarry demo build` first")
    ASSETS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cfg = QuarryConfig().model_copy(update={"memory_path": work / "memory.sqlite"})
        with open_agent(cfg, db_path=database, offline=True, memory_path=cfg.memory_path) as agent:
            port = _free_port()
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(agent, offline=True),
                    host="127.0.0.1",
                    port=port,
                    log_level="warning",
                )
            )
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            while not server.started:
                time.sleep(0.05)
            try:
                frames = _record(f"http://127.0.0.1:{port}/", work)
            finally:
                server.should_exit = True
                thread.join(timeout=10)
    first, *rest = frames
    first.save(
        ASSETS / "demo.gif",
        save_all=True,
        append_images=rest,
        duration=[1200, 900, 3800, 900, 3800, 900, 3400],
        loop=0,
        optimize=True,
    )
    print(f"wrote {ASSETS}/hero.png, follow-up.png, guard.png and demo.gif")


def _record(url: str, shots: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.goto(url)
        page.wait_for_selector(".chip")
        frames.append(_grab(page, shots, "start"))

        _ask(page, DEMO_QUESTIONS[0], frames, shots)
        page.wait_for_selector("#chart svg", timeout=20_000)
        frames.append(_grab(page, shots, "q1"))
        page.screenshot(path=str(ASSETS / "hero.png"))

        _ask(page, DEMO_QUESTIONS[1], frames, shots)
        frames.append(_grab(page, shots, "q2"))
        page.screenshot(path=str(ASSETS / "follow-up.png"))

        _ask(page, DEMO_QUESTIONS[4], frames, shots)
        frames.append(_grab(page, shots, "q3"))
        page.screenshot(path=str(ASSETS / "guard.png"))
        browser.close()
    return frames


if __name__ == "__main__":
    main()
