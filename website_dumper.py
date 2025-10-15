import asyncio
import argparse
import hashlib
import json
import os
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Set, Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse, urljoin, urldefrag

import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

# Playwright async
from playwright.async_api import async_playwright

# For robots.txt
import urllib.robotparser


# Utilities
def normalize_url(base: str, link: str) -> Optional[str]:
    if not link:
        return None
    link = link.strip()
    # ignore javascript:, mailto:, tel:, data:
    if any(link.startswith(s) for s in ("javascript:", "mailto:", "tel:", "data:")):
        return None
    joined = urljoin(base, link)
    # remove fragment
    joined, _ = urldefrag(joined)
    return joined


def same_domain(url1: str, url2: str) -> bool:
    p1 = urlparse(url1)
    p2 = urlparse(url2)
    return p1.netloc == p2.netloc


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class PageRecord:
    url: str
    status: Optional[int] = None
    title: Optional[str] = None
    meta_description: Optional[str] = None
    canonical: Optional[str] = None
    content_hash: Optional[str] = None
    text: Optional[str] = None
    links: List[str] = None
    assets: List[str] = None
    headers: Dict[str, str] = None
    retrieved_at: str = None
    fetch_mode: str = None  # 'static' or 'dynamic'
    html_path: Optional[str] = None


# Fetchers
class StaticFetcher:
    def __init__(self, session: aiohttp.ClientSession, timeout: int = 15):
        self.session = session
        self.timeout = timeout

    async def fetch(self, url: str) -> Tuple[Optional[str], Optional[int], Dict[str, str]]:
        try:
            async with self.session.get(url, timeout=self.timeout, allow_redirects=True) as resp:
                status = resp.status
                headers = dict(resp.headers)
                # attempt to read text safely
                text = await resp.text(errors='ignore')
                return text, status, headers
        except Exception as e:
            # print or log
            return None, None, {}


class DynamicFetcher:
    """
    Uses Playwright async API to render page and return rendered HTML.
    Should be used infrequently due to overhead.
    """
    def __init__(self, browser, timeout: int = 30):
        self.browser = browser
        self.timeout = timeout

    async def fetch(self, url: str) -> Tuple[Optional[str], Optional[int], Dict[str, str]]:
        page = await self.browser.new_page()
        try:
            # set reasonable viewport & userAgent
            await page.goto(url, timeout=self.timeout * 1000, wait_until="networkidle")
            content = await page.content()
            # status isn't directly available; we can use response object from goto if needed
            return content, 200, {}
        except Exception as e:
            try:
                await page.close()
            except:
                pass
            return None, None, {}
        finally:
            try:
                await page.close()
            except:
                pass



# Analyzer
def analyze_html(base_url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    desc_tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if not desc_tag:
        desc_tag = soup.find("meta", attrs={"property": re.compile("description", re.I)})
    meta_desc = desc_tag.get("content").strip() if desc_tag and desc_tag.get("content") else None
    # canonical
    can_tag = soup.find("link", rel="canonical")
    canonical = can_tag.get("href") if can_tag and can_tag.get("href") else None
    # text
    for script in soup(["script", "style", "noscript"]):
        script.extract()
    text = soup.get_text(separator=" ", strip=True)
    # links
    anchors = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        norm = normalize_url(base_url, href)
        if norm:
            anchors.add(norm)
    # assets
    assets = set()
    for tag in soup.find_all(["img", "script", "link"]):
        src = tag.get("src") or tag.get("href")
        if src:
            norm = normalize_url(base_url, src)
            if norm:
                assets.add(norm)
    return {
        "title": title,
        "meta_description": meta_desc,
        "canonical": canonical,
        "text": text,
        "links": list(anchors),
        "assets": list(assets),
    }



# Crawler
class Crawler:
    def __init__(
        self,
        start_url: str,
        *,
        max_pages: int = 500,
        max_depth: int = 3,
        concurrency: int = 5,
        dynamic_mode: str = "auto",  # 'auto'|'always'|'never'
        save_html: bool = True,
        output_dir: str = "dump_output",
        obey_robots: bool = True,
        user_agent: str = "WebsiteDumperBot/1.0 (+https://example.com/bot)",
    ):
        self.start_url = start_url
        self.parsed_start = urlparse(start_url)
        self.domain = self.parsed_start.netloc
        self.scheme = self.parsed_start.scheme
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.concurrency = concurrency
        self.dynamic_mode = dynamic_mode
        self.save_html = save_html
        self.output_dir = output_dir
        self.obey_robots = obey_robots
        self.user_agent = user_agent

        self.seen: Set[str] = set()
        self.to_visit: asyncio.Queue = asyncio.Queue()
        self.records: Dict[str, PageRecord] = {}
        self.robot_parser = urllib.robotparser.RobotFileParser()
        self.rate_delay = 0

    def is_allowed_by_robots(self, url: str) -> bool:
        if not self.obey_robots:
            return True
        try:
            return self.robot_parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    async def init_robots(self):
        try:
            robots_url = f"{self.scheme}://{self.domain}/robots.txt"
            rp = self.robot_parser
            rp.set_url(robots_url)
            rp.read()
            # crawl delay not standardized; try reading from robots.txt content
            # urllib doesn't expose Crawl-delay parsing; naive parse:
            try:
                text = ""
                async with aiohttp.ClientSession() as s:
                    async with s.get(robots_url, timeout=10) as r:
                        text = await r.text()
                m = re.search(r"Crawl-delay:\s*(\d+)", text, re.I)
                if m:
                    self.rate_delay = int(m.group(1))
            except Exception:
                pass
        except Exception:
            pass

    async def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        await self.init_robots()
        start = time.time()
        # push start
        await self.to_visit.put((self.start_url, 0))
        self.seen.add(self.start_url)

        # aiohttp session for static fetches
        headers = {"User-Agent": self.user_agent}
        conn = aiohttp.TCPConnector(limit=None, force_close=True)
        async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
            static_fetcher = StaticFetcher(session)
            # start Playwright for dynamic if needed
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            dynamic_fetcher = DynamicFetcher(browser)

            # worker tasks
            workers = [asyncio.create_task(self.worker(static_fetcher, dynamic_fetcher)) for _ in range(self.concurrency)]
            await self.to_visit.join()
            for w in workers:
                w.cancel()

            await browser.close()
            await playwright.stop()
        elapsed = time.time() - start
        return elapsed

    async def worker(self, static_fetcher: StaticFetcher, dynamic_fetcher: DynamicFetcher):
        while True:
            try:
                url, depth = await self.to_visit.get()
            except asyncio.CancelledError:
                return
            try:
                # respect robots
                if not self.is_allowed_by_robots(url):
                    self.to_visit.task_done()
                    continue
                # fetch (choose static or dynamic)
                use_dynamic = False
                if self.dynamic_mode == "always":
                    use_dynamic = True
                elif self.dynamic_mode == "never":
                    use_dynamic = False
                else:  # auto: heuristic
                    # simple heuristic: if URL contains query params or looks like a route, use static first, then dynamic if extracted bodies are too small
                    use_dynamic = False

                html, status, headers = await static_fetcher.fetch(url)
                fetch_mode = "static"
                if (not html or len(html) < 200) and (self.dynamic_mode in ("auto", "always")):
                    # try dynamic
                    dyn_html, dyn_status, dyn_headers = await dynamic_fetcher.fetch(url)
                    if dyn_html:
                        html = dyn_html
                        status = dyn_status or status
                        headers = dyn_headers or headers
                        fetch_mode = "dynamic"

                if html is None:
                    # record failure
                    rec = PageRecord(
                        url=url,
                        status=status,
                        title=None,
                        meta_description=None,
                        canonical=None,
                        content_hash=None,
                        text=None,
                        links=[],
                        assets=[],
                        headers=headers,
                        retrieved_at=datetime.utcnow().isoformat() + "Z",
                        fetch_mode=fetch_mode,
                        html_path=None,
                    )
                    self.records[url] = rec
                    self.to_visit.task_done()
                    continue

                # analyze
                info = analyze_html(url, html)
                content_hash = compute_hash(html)
                # save html to disk
                html_path = None
                if self.save_html:
                    safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', urlparse(url).path or 'root')
                    if len(safe_name) > 200:
                        safe_name = safe_name[:200]
                    filename = hashlib.sha1(url.encode()).hexdigest() + ".html"
                    html_path = os.path.join(self.output_dir, filename)
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)

                rec = PageRecord(
                    url=url,
                    status=status,
                    title=info.get("title"),
                    meta_description=info.get("meta_description"),
                    canonical=info.get("canonical"),
                    content_hash=content_hash,
                    text=(info.get("text")[:10000] if info.get("text") else None),  # truncate text
                    links=info.get("links"),
                    assets=info.get("assets"),
                    headers=headers,
                    retrieved_at=datetime.utcnow().isoformat() + "Z",
                    fetch_mode=fetch_mode,
                    html_path=html_path,
                )
                self.records[url] = rec

                # enqueue neighbors
                if depth < self.max_depth and len(self.records) < self.max_pages:
                    for link in info.get("links", []):
                        if same_domain(self.start_url, link):
                            if link not in self.seen:
                                self.seen.add(link)
                                await self.to_visit.put((link, depth + 1))
                # polite delay
                if self.rate_delay > 0:
                    await asyncio.sleep(self.rate_delay)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                # continue on error
                pass
            finally:
                self.to_visit.task_done()

    def dump_json(self, path: str):
        out = {
            "start_url": self.start_url,
            "domain": self.domain,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "page_count": len(self.records),
            "pages": [asdict(rec) for rec in self.records.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)



# CLI
def main():
    parser = argparse.ArgumentParser(description="Website Dumper - crawl and dump pages to JSON")
    parser.add_argument("url", help="Start URL (e.g. https://example.com)")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dynamic-mode", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--output", default="site_dump.json")
    parser.add_argument("--output-dir", default="dump_output")
    parser.add_argument("--no-html", action="store_true", help="Do not save raw HTML files")
    args = parser.parse_args()

    c = Crawler(
        args.url,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        concurrency=args.concurrency,
        dynamic_mode=args.dynamic_mode,
        save_html=not args.no_html,
        output_dir=args.output_dir,
    )

    loop = asyncio.get_event_loop()
    try:
        elapsed = loop.run_until_complete(c.run())
        c.dump_json(args.output)
        print(f"Done. Crawled {len(c.records)} pages in {elapsed:.1f}s. Output: {args.output}")
    finally:
        # close loop
        try:
            loop.close()
        except:
            pass


if __name__ == "__main__":
    main()

