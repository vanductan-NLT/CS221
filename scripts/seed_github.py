"""Tạo nhãn, milestone, issue (và một GitHub Project) từ scripts/seed/issues.json.

Chỉ chạy MỘT lần, trên repo chưa có issue nào, để số issue khớp với số ghi trong nội dung
(issue thứ 1 trong file là #1, thứ 2 là #2, ...).

Cần: GitHub CLI `gh` đã đăng nhập, đứng trong thư mục repo đã có remote trên GitHub.
Muốn tạo Project thì token cần quyền project:  gh auth refresh -s project

Chạy:
    python scripts/seed_github.py --dry-run                         chỉ in ra sẽ tạo gì
    python scripts/seed_github.py                                   tạo nhãn, milestone, issue
    python scripts/seed_github.py --project "CS221 · Đồ án PopQA"   như trên, thêm Project chứa mọi issue
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "scripts" / "seed" / "issues.json"
DRY = "--dry-run" in sys.argv
PROJECT = None
if "--project" in sys.argv:
    PROJECT = sys.argv[sys.argv.index("--project") + 1]


def gh(*args, capture=False):
    cmd = ["gh", *args]
    if DRY:
        print("[dry-run]", " ".join(a if len(a) < 60 else a[:57] + "..." for a in cmd))
        return ""
    res = subprocess.run(cmd, check=True, text=True, capture_output=capture, encoding="utf-8")
    return res.stdout if capture else ""


def main():
    data = json.loads(SEED.read_text(encoding="utf-8"))

    repo = ""
    if not DRY:
        repo = gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner", capture=True).strip()
        existing = gh("issue", "list", "--state", "all", "--limit", "1", "--json", "number", capture=True)
        if json.loads(existing or "[]"):
            sys.exit("Repo đã có issue. Dừng lại để số issue không bị lệch.")

    for lb in data["labels"]:
        gh("label", "create", lb["name"], "--color", lb["color"], "--description", lb["description"], "--force")

    for ms in data["milestones"]:
        gh("api", "repos/{owner}/{repo}/milestones", "-f", "title=" + ms["title"],
           "-f", "description=" + ms["description"], "--silent")

    urls = []
    for i, it in enumerate(data["issues"], 1):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(it["body"])
            body_file = f.name
        args = ["issue", "create", "--title", it["title"], "--body-file", body_file, "--milestone", it["milestone"]]
        for lb in it["labels"]:
            args += ["--label", lb]
        url = gh(*args, capture=True).strip()
        urls.append(url)
        print(f"#{i} {it['title']}")

    if PROJECT:
        out = gh("project", "create", "--owner", "@me", "--title", PROJECT, "--format", "json", capture=True)
        number = str(json.loads(out)["number"]) if out else "<số project>"
        if repo:
            gh("project", "link", number, "--owner", "@me", "--repo", repo)
        for url in urls:
            gh("project", "item-add", number, "--owner", "@me", "--url", url or "<url issue>")
        print(f"Project '{PROJECT}' (số {number}) đã có {len(urls)} issue.")


if __name__ == "__main__":
    main()
