"""Save official club news responses for review; no inferred completion."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from html import unescape

OUT = Path(r"G:\Codex\football-prediction-2026-10-22\data\prematch-intelligence\20261024_batch\result_observation_20261024T160147143875Z")

def fetch(item):
    label, url = item
    start = datetime.now(timezone.utc).isoformat()
    try:
        with urlopen(Request(url, headers={"User-Agent": "FootballResearchPersonal/0.30"}), timeout=25) as response:
            body = response.read()
            meta = {"source_url": url, "response_url": response.url, "status": response.status,
                    "request_started_at": start, "received_at": datetime.now(timezone.utc).isoformat(),
                    "response_date": response.headers.get("Date"), "age_seconds": response.headers.get("Age"),
                    "content_type": response.headers.get("Content-Type"), "bytes": len(body)}
    except Exception as exc:
        body = exc.read() if isinstance(exc, HTTPError) else b""
        meta = {"source_url": url, "status": getattr(exc, "code", None), "request_started_at": start,
                "received_at": datetime.now(timezone.utc).isoformat(), "error": str(exc), "bytes": len(body)}
    (OUT / f"{label}_news_raw.html").write_bytes(body)
    (OUT / f"{label}_news_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    content = body.decode("utf-8", errors="replace")
    links = sorted(set(unescape(x.replace('\\/', '/')) for x in re.findall(r'''(?:href=)["']([^"']+)["']''', content)))
    relevant = [x for x in links if any(y in x.lower() for y in ("everton", "arsenal", "coventry", "fulham", "report"))]
    return {"label": label, "metadata": meta, "relevant_links": relevant[:45], "script_count": content.count("<script")}

if __name__ == "__main__":
    sources = [("fulham_home", "https://www.fulhamfc.com/"),
               ("coventry_home", "https://www.ccfc.co.uk/")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, sources):
            print(json.dumps(result, ensure_ascii=False))
    for label in ("fulham_home", "coventry_home"):
        content = (OUT / f"{label}_news_raw.html").read_text(encoding="utf-8")
        for pattern in (r".{0,100}Coventry.{0,250}", r".{0,100}defeat.{0,250}"):
            matches = [m.group() for m in re.finditer(pattern, content, re.I)]
            print(json.dumps({"label": label, "pattern": pattern, "matches": matches[:16]}, ensure_ascii=False))
