"""Check built pages have the shared theme, return navigation and local assets."""

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.back = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "library-return" in attrs.get("class", "").split():
            self.back = (
                attrs.get("href")
                == "https://nz-gravity.github.io/LogPSplinePSD/intro.html"
            )
        attribute = {"a": "href", "img": "src", "script": "src", "link": "href"}.get(
            tag
        )
        if attribute and attribute in attrs:
            self.links.append(attrs[attribute])


root = Path(sys.argv[1]).resolve()
pages = [
    root / f"{name}.html"
    for name in (
        "intro",
        "running",
        "notebooks/01_generate_data",
        "notebooks/02_time_varying_psd",
        "notebooks/03_xyz_psd",
        "notebooks/04_short_xyz_gaps",
        "notebook_models",
        "gaplike_xyz",
    )
]
for page in pages:
    html = page.read_text()
    parsed = Links()
    parsed.feed(html)
    assert parsed.back, f"Missing library return button: {page}"
    assert "styles/sphinx-book-theme.css" in html, f"Wrong theme: {page}"
    assert "case-studies.css" in html, f"Missing navigation styling: {page}"
    for link in parsed.links:
        target = urlsplit(link)
        if target.scheme or target.netloc or not target.path:
            continue
        assert (page.parent / unquote(target.path)).exists(), (
            f"Broken local link in {page}: {link}"
        )
print(f"Verified theme, return navigation and local links on {len(pages)} pages")
