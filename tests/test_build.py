from __future__ import annotations
from pathlib import Path
import tempfile
import unittest

from build import parse_note, render_md


class ContentTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "note.md"
        path.write_text(text, encoding="utf-8")
        return path

    def valid(self, changes: str = "") -> str:
        return (
            '---\ntitle: "Test"\nslug: "test-note"\nkind: "paper"\n'
            'date: "2026-09-18"\nsummary: "Test summary"\ntags: ["test"]\n'
            'published: true\n' + changes + '---\n## Heading\n\nBody.\n'
        )

    def test_valid_note(self):
        note = parse_note(self.write(self.valid()))
        self.assertEqual(note["url"], "notes/test-note.html")
        self.assertEqual(note["date"], "2026-09-18")
        self.assertIn('id="section-1"', note["body_html"])

    def test_draft_excluded(self):
        self.assertEqual(
            parse_note(self.write('---\npublished: false\n---\nPrivate draft.')),
            {"published": False},
        )

    def test_missing_publish_is_excluded(self):
        self.assertFalse(parse_note(self.write('---\ntitle: "Draft"\n---\nText'))["published"])

    def test_bad_slug(self):
        with self.assertRaises(ValueError):
            parse_note(self.write(self.valid().replace("test-note", "../outside")))

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            parse_note(self.write(self.valid().replace('"paper"', '"unknown"')))

    def test_tags_must_be_list(self):
        with self.assertRaises(ValueError):
            parse_note(self.write(self.valid().replace('["test"]', '"test"')))

    def test_raw_html_escaped(self):
        html, _ = render_md('<script>alert(1)</script>')
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_unsafe_link_blocked(self):
        html, _ = render_md("[test](javascript:alert%281%29)")
        self.assertNotIn('href="javascript:', html)

    def test_table(self):
        html, _ = render_md("| A | B |\n|---|---|\n| 1 | 2 |")
        self.assertIn("<table>", html)

    def test_headings_are_unique(self):
        _, toc = render_md("## Same\n\n## Same")
        self.assertEqual([h["id"] for h in toc], ["section-1", "section-2"])


if __name__ == "__main__":
    unittest.main()
