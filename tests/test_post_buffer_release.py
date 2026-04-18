from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import post_buffer_release


def test_build_release_post_text_includes_successful_platforms_and_url() -> None:
    text = post_buffer_release.build_release_post_text(
        app_name="Face-Local",
        tag="v1.2.3",
        release_url="https://github.com/example/face-local/releases/tag/v1.2.3",
        successful_platforms=["macOS", "Windows"],
    )

    assert "Face-Local scans photos offline, finds faces" in text
    assert "A Face-Local helyben felismeri és csoportosítja az arcokat" in text
    assert "https://github.com/example/face-local/releases/tag/v1.2.3" in text
    assert len(text) <= post_buffer_release.MAX_POST_LENGTH


def test_select_target_channel_prefers_named_match() -> None:
    class StubClient:
        def get_organization_ids(self) -> list[str]:
            return ["org-1"]

        def get_channels(self, organization_id: str):
            assert organization_id == "org-1"
            return [
                {"id": "1", "name": "@other", "displayName": "Other", "service": "twitter"},
                {"id": "2", "name": "@ben", "displayName": "BenKoncsik", "service": "twitter"},
            ]

    selected = post_buffer_release.select_target_channel(
        client=StubClient(),
        organization_id=None,
        channel_id=None,
        channel_name="Ben",
        channel_service="twitter",
    )

    assert selected["id"] == "2"


def test_empty_buffer_post_mode_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("BUFFER_POST_MODE", "")

    mode = post_buffer_release.validate_mode((__import__("os").environ.get("BUFFER_POST_MODE") or "").strip() or "shareNow")

    assert mode == "shareNow"
