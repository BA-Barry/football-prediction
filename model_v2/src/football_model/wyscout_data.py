"""Official bounded Wyscout public data retrieval and low-memory JSON arrays.

Only England is unpacked from the author's public archives. Original ZIPs and
source article/license metadata are retained. No package downgrade, account,
paid API or generated xG is involved.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .statsbomb_data import write_json


ITEMS = ((7770422, 14464622, "matches.zip"), (7770599, 14464685, "events.zip"),
         (7765196, 15073721, "players.json"), (7765310, 15073697, "teams.json"),
         (11743818, 21385239, "tags2name.csv"), (11743836, 21385245, "eventid2name.csv"))


def fetch(url, destination):
    destination = Path(destination)
    retrieved = None
    reused = destination.exists()
    if not reused:
        temporary = destination.with_name(destination.name + ".partial-" + datetime.now(timezone.utc).strftime("%H%M%S%f"))
        with urlopen(Request(url, headers={"User-Agent": "personal-football-statistics/0.8"}), timeout=60) as response:
            with temporary.open("xb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        temporary.rename(destination)
        retrieved = datetime.now(timezone.utc).isoformat()
    return {"source_url": url, "local_path": str(destination), "bytes": destination.stat().st_size,
            "reused": reused, "retrieved_at_utc": retrieved,
            "observed_local_at_utc": datetime.now(timezone.utc).isoformat(), "historical_first_available_at": None}


def unpack_member(archive, filename, destination):
    """Select one named member, never extract archive-supplied paths."""
    destination = Path(destination)
    with ZipFile(archive) as zipped:
        candidates = [info for info in zipped.infolist() if not info.is_dir()
                      and Path(info.filename.replace("\\", "/")).name == filename
                      and "__MACOSX" not in info.filename]
        if len(candidates) != 1:
            raise ValueError(f"Need one unambiguous source member {filename}")
        info = candidates[0]
        if not destination.exists():
            with zipped.open(info) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        else:
            # A same-length unrelated file is not a valid cached extraction.
            # Compare to the retained archive directly; no extra digest files.
            with zipped.open(info) as source, destination.open("rb") as target:
                while True:
                    expected, actual = source.read(1024 * 1024), target.read(1024 * 1024)
                    if expected != actual:
                        raise ValueError("Existing extraction content differs from archive member")
                    if not expected:
                        break
        if destination.stat().st_size != info.file_size:
            raise ValueError("Unpacked source size does not match archive member")
        return {"archive": str(archive), "member": info.filename, "local_path": str(destination),
                "bytes": info.file_size, "raw_content_changed": False}


def iter_json_array(path, chunk_size=65536):
    """Stream original JSON objects; no full event-season memory allocation."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Positive integer JSON chunk size required")
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, finished, started, need_value, yielded_any = "", False, False, True, False
        while True:
            buffer = buffer.lstrip()
            if not started:
                if not buffer and not finished:
                    chunk = stream.read(chunk_size)
                    buffer += chunk
                    finished = not chunk
                    continue
                if not buffer.startswith("["):
                    raise ValueError("Expected a JSON array")
                buffer, started = buffer[1:], True
                continue
            if buffer.startswith("]"):
                if need_value and yielded_any:
                    raise ValueError("Trailing JSON array comma")
                remainder = buffer[1:] + stream.read()
                if remainder.strip():
                    raise ValueError("Trailing content after JSON array")
                return
            if not buffer:
                if finished:
                    raise ValueError("Unterminated JSON array")
                chunk = stream.read(chunk_size)
                buffer += chunk
                finished = not chunk
                continue
            if not need_value:
                if buffer[0] != ",":
                    raise ValueError("Missing JSON array separator")
                buffer, need_value = buffer[1:], True
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if finished:
                    raise
                chunk = stream.read(chunk_size)
                buffer += chunk
                finished = not chunk
                continue
            # Events are dictionaries: rejecting scalar splits avoids accepting
            # a partial numeric token at a chunk boundary.
            if not isinstance(value, dict):
                raise ValueError("Expected JSON object array elements")
            yield value
            yielded_any = True
            buffer, need_value = buffer[end:], False


def iter_match_events(path):
    current, group, closed = None, [], set()
    for event in iter_json_array(path):
        mid = event.get("matchId")
        if type(mid) is not int or mid <= 0:
            raise ValueError("Missing positive event match ID")
        if current is not None and mid != current:
            yield current, group
            closed.add(current)
            group = []
        if mid in closed:
            raise ValueError("Source match events are not contiguous; explicit regrouping needed")
        current = mid
        group.append(event)
    if current is not None:
        yield current, group


def download_public(root):
    destination = Path(root).resolve() / "data/wyscout/premier-league-2017-18"
    destination.mkdir(parents=True, exist_ok=True)
    records, articles = [], []
    for article_id, file_id, filename in ITEMS:
        article_path = destination / f"article_{article_id}.json"
        records.append(fetch(f"https://api.figshare.com/v2/articles/{article_id}", article_path))
        article = json.loads(article_path.read_text(encoding="utf-8"))
        source_file = next(f for f in article["files"] if f["id"] == file_id)
        articles.append({"id": article_id, "title": article["title"], "doi": article.get("doi"),
                         "authors": article.get("authors"), "license": article["license"],
                         "file_id": file_id, "file_name": source_file["name"]})
        record = fetch(source_file["download_url"], destination / filename)
        if record["bytes"] != source_file["size"]:
            raise ValueError(f"Official file size disagreement: {filename}")
        records.append(record)
        print(f"Official public source ready: {filename}, {record['bytes']} bytes", flush=True)
    extracted = [unpack_member(destination / "matches.zip", "matches_England.json", destination / "matches_England.json"),
                 unpack_member(destination / "events.zip", "events_England.json", destination / "events_England.json")]
    fd_path = destination / "Football-Data_E0_2017-18.csv"
    records.append(fetch("https://www.football-data.co.uk/mmz4281/1718/E0.csv", fd_path))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(destination / f"下载来源_{stamp}.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "articles": articles,
        "files": records, "selected_archive_members": extracted,
        "collection": "https://figshare.com/collections/Soccer_match_event_dataset/4415000",
        "paper_doi": "10.1038/s41597-019-0247-7", "native_xg": None,
        "purpose": "personal noncommercial retrospective football statistics/entertainment",
        "football_data_used_for": "independent date/team/score reconciliation; no odds in this experiment",
        "registration_or_paid_api": False, "raw_data_redistribution": False})
    print(f"Public England source ready: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    download_public(parser.parse_args().root)
