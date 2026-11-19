"""One-off official CSV download requested for personal, noncommercial viewing.

No model imports, fitting, scraping of other pages, mirrors, or retry bypasses.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = REPOSITORY_ROOT
FILES = (("E0", "英超"), ("D1", "德甲"), ("SP1", "西甲"), ("I1", "意甲"), ("F1", "法甲"))


def main():
    started = datetime.now(timezone.utc)
    destination = ROOT / "outputs" / ("Football-Data_2024-25_个人使用_" + started.strftime("%Y%m%dT%H%M%SZ"))
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"purpose": "Personal noncommercial football-model viewing and entertainment; no training in this download task",
                "started_at": started.isoformat(), "season": "2024/25", "directory": str(destination),
                "source_information": "https://football-data.co.uk/data.php", "files": [], "error": None}
    for code, league in FILES:
        url = f"https://football-data.co.uk/mmz4281/2425/{code}.csv"
        try:
            request = Request(url, headers={"User-Agent": "Codex-personal-file-download/1.0"})
            with urlopen(request, timeout=30) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
                resolved_url = response.url
            # Check only file identity, not match or odds content. A block page
            # must not be reported as a successful CSV download.
            first_line = body.splitlines()[0] if body else b""
            if not first_line.startswith((b"Div,", b"\xef\xbb\xbfDiv,")):
                raise ValueError("Response is not the expected CSV header; file not saved")
            file_path = destination / f"{code}.csv"
            file_path.write_bytes(body)
            manifest["files"].append({"league": league, "path": str(file_path), "source": url,
                                       "resolved_url": resolved_url, "bytes": len(body),
                                       "content_type": content_type,
                                       "downloaded_at": datetime.now(timezone.utc).isoformat()})
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            manifest["error"] = {"source": url, "message": str(error),
                                 "retry_after": error.headers.get("Retry-After") if isinstance(error, HTTPError) else None}
            # Stop on server refusal/rate limiting; do not rotate hostnames or use mirrors.
            break
    (destination / "下载来源.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Football-Data：个人非商业下载", "",
             "用途：个人查看足球预测模型与娱乐；本次只下载官方公开文件，没有训练模型。", "",
             "赛季：2024/25。以下保留原始CSV，不修改内容。", "",
             "| 联赛 | 文件 | 大小（字节） |", "|---|---|---:|"]
    for item in manifest["files"]:
        lines.append(f"| {item['league']} | [{Path(item['path']).name}]({Path(item['path']).name}) | {item['bytes']} |")
    if manifest["error"]:
        lines += ["", "下载未全部完成：" + manifest["error"]["message"] + "。未绕过访问限制。"]
    lines += ["", "来源：[官方数据说明](https://football-data.co.uk/data.php)、[字段说明](https://football-data.co.uk/notes.txt)。", "",
              "赔率列包含不同采集阶段；不将pre-closing统一称为开盘，也不将文件下载时间当作历史报价时间。", "",
              "本次用户用途说明不代表第三方额外授权。未启动模型训练、未将文件并入现有训练输入、未对外发布。"]
    (destination / "下载说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(destination), "downloaded": len(manifest["files"]),
                      "files": [{"league": row["league"], "path": row["path"], "bytes": row["bytes"]} for row in manifest["files"]],
                      "error": manifest["error"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
