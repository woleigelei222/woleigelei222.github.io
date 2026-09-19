"""Build a small Markdown learning portfolio. Python 3.11+.

Only content/*.md with published: true are published.
This is not access control: files in a public source repository are still public.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import unquote, urlsplit

import jinja2
import mistune
import yaml

ROOT = Path(__file__).resolve().parent
KINDS = {
    "paper": {"label": "论文笔记", "short": "论文", "index": "papers.html"},
    "algorithm": {"label": "算法理解", "short": "算法", "index": "algorithms.html"},
    "project": {"label": "项目实践", "short": "项目", "index": "projects.html"},
    "journal": {"label": "学习日志", "short": "日志", "index": "journal.html"},
}
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class NoteRenderer(mistune.HTMLRenderer):
    """Deterministic heading IDs for a table of contents."""
    def __init__(self):
        super().__init__(escape=True)
        self.headings: list[dict] = []

    def heading(self, text: str, level: int, **attrs) -> str:
        heading_id = f"section-{len(self.headings) + 1}"
        self.headings.append({"id": heading_id, "level": level, "label": text})
        return f'<h{level} id="{heading_id}">{text}</h{level}>\n'


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if (tag, name) in {("a", "href"), ("img", "src"), ("link", "href"), ("script", "src")} and value:
                self.links.append(value)


def read_yaml(path: Path) -> dict:
    obj = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return obj


def render_md(text: str) -> tuple[str, list[dict]]:
    renderer = NoteRenderer()
    md = mistune.create_markdown(renderer=renderer, plugins=["table", "footnotes", "strikethrough"])
    return md(text), renderer.headings


def parse_note(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)\Z", text, re.S)
    if not match:
        raise ValueError(f"{path}: add YAML front matter between --- lines")
    meta = yaml.safe_load(match.group(1))
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: front matter must be a YAML mapping")
    # Draft templates may have incomplete fields; never publish by default.
    if meta.get("published") is not True:
        return {"published": False}
    for key in ["title", "slug", "kind", "date", "summary"]:
        if key not in meta:
            raise ValueError(f"{path}: missing {key!r}")
    if not isinstance(meta["slug"], str) or not SLUG.fullmatch(meta["slug"]):
        raise ValueError(f"{path}: slug must use lowercase letters, numbers and hyphens")
    if meta["kind"] not in KINDS:
        raise ValueError(f"{path}: kind must be one of {list(KINDS)}")
    value = meta["date"]
    if isinstance(value, datetime):
        value = value.date()
    elif not isinstance(value, date):
        value = date.fromisoformat(str(value))
    meta["date"] = value.isoformat()
    for key in ("title", "summary"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise ValueError(f"{path}: {key} must be nonempty text")
    tags = meta.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValueError(f"{path}: tags must be a list of strings")
    for key in ["featured", "example"]:
        if key in meta and not isinstance(meta[key], bool):
            raise ValueError(f"{path}: {key} must be true or false")
    html, toc = render_md(match.group(2))
    meta.update({
        "published": True,
        "tags": tags,
        "featured": meta.get("featured", False),
        "example": meta.get("example", False),
        "body_html": html,
        "toc": toc,
        "url": f"notes/{meta['slug']}.html",
        "kind_label": KINDS[meta["kind"]]["label"],
        "kind_index": KINDS[meta["kind"]]["index"],
        "search": " ".join([meta["title"], meta["summary"], *tags]).lower(),
    })
    return meta


def check_site(output: Path) -> None:
    broken = []
    for page in output.rglob("*.html"):
        parser = LocalLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        for href in parser.links:
            if href.startswith(("#", "//", "data:")):
                continue
            u = urlsplit(href)
            if u.scheme:
                continue
            target = (page.parent / unquote(u.path)).resolve()
            if not u.path:
                continue
            if not target.is_relative_to(output.resolve()) or not target.exists():
                broken.append(f"{page.relative_to(output)} -> {href}")
    if broken:
        raise ValueError("Broken local links:\n" + "\n".join(broken))


def build() -> None:
    config = read_yaml(ROOT / "site.yml")
    for key in ["title", "author", "role", "description"]:
        if not isinstance(config.get(key), str):
            raise ValueError(f"site.yml: {key} must be text")
    for key in ["github", "resume"]:
        value = config.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"site.yml: {key} must be text")
        if value:
            parsed = urlsplit(value)
            if key == "github" and (parsed.scheme != "https" or not parsed.netloc):
                raise ValueError("site.yml: github must be a full https URL")
            if key == "resume":
                safe_local = not parsed.scheme and not value.startswith(("/", "\\")) and ".." not in Path(value).parts
                if not ((parsed.scheme == "https" and parsed.netloc) or safe_local):
                    raise ValueError("site.yml: resume must be an https URL or a relative file path")
    email = config.get("email", "")
    if not isinstance(email, str) or (email and not re.fullmatch(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+", email)):
        raise ValueError("site.yml: email must be blank or a valid public contact address")

    notes = [parse_note(p) for p in sorted((ROOT / "content").glob("*.md"))]
    notes = [n for n in notes if n["published"]]
    if not config.get("show_examples", True):
        notes = [n for n in notes if not n["example"]]
    slugs = [n["slug"] for n in notes]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Two published notes have the same slug")
    notes.sort(key=lambda n: (n["date"], n["slug"]), reverse=True)
    public_notes = [n for n in notes if not n["example"]]
    counts = {k: sum(n["kind"] == k for n in public_notes) for k in KINDS}
    selected = [n for n in notes if n["featured"]][:3]
    output = ROOT / "public"
    if output.is_symlink():
        raise ValueError("Refusing to replace a symlink at public/")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    shutil.copytree(ROOT / "assets", output / "assets")
    (output / "templates").mkdir()
    for path in (ROOT / "note-templates").glob("*.md"):
        shutil.copy2(path, output / "templates" / path.name)
    (output / ".nojekyll").touch()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(ROOT / "layouts"),
        autoescape=jinja2.select_autoescape(["html"]),
        undefined=jinja2.StrictUndefined,
    )
    common = dict(site=config, kinds=KINDS, notes=notes, selected=selected,
                  counts=counts, real_count=len(public_notes), example_count=len(notes)-len(public_notes),
                  all_tags=sorted({tag for n in notes for tag in n["tags"]}))
    def write_page(path: str, template: str, **params) -> None:
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        prefix = "../" * (len(Path(path).parts) - 1)
        destination.write_text(env.get_template(template).render(
            **common, prefix=prefix, **params), encoding="utf-8")

    write_page("index.html", "home.html", active="home", title="首页")
    write_page("notes.html", "archive.html", active="notes", title="全部记录",
               entries=notes, intro="把零散的阅读、推导与实验，整理成可以回看的知识。", filterable=True)
    for kind, item in KINDS.items():
        intro = {
            "paper": "先理解作者解决了什么，再记录我认同什么、仍然怀疑什么。",
            "algorithm": "不只记住步骤，也用例子、反例和代码检查自己的理解。",
            "project": "从一个可验证的问题出发，留下环境、代码、结果与局限。",
            "journal": "记录今天真正弄懂的一件事，以及下一次要验证的问题。",
        }[kind]
        write_page(item["index"], "archive.html", active=kind, title=item["label"],
                   entries=[n for n in notes if n["kind"] == kind], intro=intro, filterable=False)
    for n in notes:
        write_page(n["url"], "post.html", active=n["kind"], title=n["title"], note=n)
    for page in ["about", "guide"]:
        html, toc = render_md((ROOT / "pages" / f"{page}.md").read_text(encoding="utf-8"))
        write_page(f"{page}.html", "page.html", active=page,
                   title={"about": "关于我", "guide": "写作指南"}[page],
                   body_html=html, toc=toc)
    check_site(output)
    print(f"Built {len(list(output.rglob('*.html')))} HTML pages; {len(public_notes)} real notes, "
          f"{len(notes)-len(public_notes)} examples. All local links checked.")
    print(f"Preview: {(output / 'index.html').as_uri()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        build()
    except (ValueError, OSError, yaml.YAMLError, jinja2.TemplateError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        sys.exit(1)
