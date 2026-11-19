"""One bounded official-site refresh; stores source responses without settlement inference."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

OUT = Path(r"G:\Codex\football-prediction-2026-10-22\data\prematch-intelligence\20261024_batch\result_observation_20261024T163124853784Z")

def fetch(item):
    name, url = item
    meta = {"source_url": url, "request_started_at": datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url, headers={"User-Agent": "FootballResearchPersonal/0.30"}), timeout=25) as response:
            body = response.read()
            meta.update(status=response.status, response_url=response.url, response_date=response.headers.get("Date"),
                        age_seconds=response.headers.get("Age"), content_type=response.headers.get("Content-Type"))
    except Exception as exc:
        body = exc.read() if isinstance(exc, HTTPError) else b""
        meta.update(status=getattr(exc, "code", None), error=str(exc))
    meta.update(received_at=datetime.now(timezone.utc).isoformat(), bytes=len(body), raw_path=f"{name}_raw.txt")
    (OUT / meta["raw_path"]).write_bytes(body)
    (OUT / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    content = body.decode("utf-8", errors="replace")
    if name.endswith("client"):
        snippets = [m.group() for m in re.finditer(r".{0,160}(?:NEWSAPI|newsApi|newsapi|/articles|/news/|article-by-slug|articleSlug).{0,200}", content)]
        return {"metadata": meta, "snippets": snippets[:30]}
    page = BeautifulSoup(content, "html.parser")
    return {"metadata": meta, "title": page.title.get_text() if page.title else None,
            "page_text": page.get_text(" ", strip=True)[:13000]}

if __name__ == "__main__":
    sources = [
        ("fulham_report", "https://www.fulhamfc.com/news/2026/october/24/coventry-0-1-fulham/"),
        ("coventry_report", "https://www.ccfc.co.uk/news/2026/october/24/report--coventry-city-0-1-fulham/"),
        ("fulham_client", "https://www.fulhamfc.com/_nuxt/bJa5KdxS.js")
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, sources):
            print(json.dumps(result, ensure_ascii=False), flush=True)
