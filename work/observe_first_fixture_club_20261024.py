"""Supplement the FPL observation with one official club match-report receipt."""
from html.parser import HTMLParser
import json
from pathlib import Path

from observe_batch_results_20261024 import download, stamp, write_json


class HeadingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture = None
        self.values = []

    def handle_starttag(self, tag, attrs):
        if tag in {"title", "h1"}:
            self.capture = tag

    def handle_endtag(self, tag):
        if tag == self.capture:
            self.capture = None

    def handle_data(self, data):
        if self.capture and data.strip():
            self.values.append({"tag": self.capture, "text": data.strip()})


def main():
    output_dir = Path(r"G:\Codex\football-prediction-2026-10-22\data\prematch-intelligence\20261024_batch\result_observation_20261024T031009082510Z")
    label, body, metadata = download(("official_club_report", "https://www.itfc.co.uk/news/2026/october/23/town-1-nottingham-forest-1/"))
    if body:
        metadata["raw_path"] = "official_club_report_raw.html" if metadata.get("status") == 200 else "official_club_report_error_body.bin"
        with (output_dir / metadata["raw_path"]).open("xb") as output:
            output.write(body)
    parser = HeadingParser()
    if metadata.get("status") == 200:
        parser.feed(body.decode("utf-8", errors="replace"))
    report = {
        "recorded_at": stamp(),
        "source_metadata": metadata,
        "extracted_headings": parser.values,
        "web_tool_observation": {
            "source_url": "https://www.itfc.co.uk/news/2026/october/23/town-1-nottingham-forest-1/",
            "observed_at": "2026-10-24T03:10:27Z",
            "page_category": "Match report",
            "heading": "TOWN 1 NOTTINGHAM FOREST 1",
            "page_display_date": "23 Oct 2026",
            "limitation": "Public page heading verified; full article body not rendered by web tool. This is supplementary official evidence, distinct from FPL finished=false / finished_provisional=true flags.",
        },
        "forecast_changes": 0,
        "model_fits": 0,
    }
    write_json(output_dir / "official_club_report_observation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
