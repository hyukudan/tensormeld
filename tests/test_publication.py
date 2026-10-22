from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicTransparencyGuards(unittest.TestCase):
    def test_publication_decision_is_recorded(self):
        adr = (ROOT / "docs" / "decisions" / "0010-public-transparency-publication.md").read_text(encoding="utf-8")
        self.assertIn("publicly while the project is", adr)
        self.assertIn("still pre-alpha", adr)
        self.assertIn("engineering transparency", adr)
        self.assertIn("production readiness", adr)

    def test_status_keeps_publication_separate_from_qualification(self):
        status = (ROOT / "docs" / "implementation" / "status.md").read_text(encoding="utf-8")
        self.assertIn("public for engineering transparency", status)
        self.assertIn("does not change", status)
        self.assertIn("pre-alpha", status)

    def test_obsolete_private_publication_helper_is_absent(self):
        self.assertFalse((ROOT / "scripts" / "publish_private.py").exists())

    def test_third_party_notice_defers_to_root_license(self):
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("repository root `LICENSE`", notice)


if __name__ == "__main__":
    unittest.main()
