from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import post_x_release


def test_build_release_post_text_includes_successful_platforms_and_url() -> None:
    text = post_x_release.build_release_post_text(
        app_name="Face-Local",
        tag="v1.2.3",
        release_url="https://github.com/example/face-local/releases/tag/v1.2.3",
        successful_platforms=["macOS", "Windows"],
    )

    assert "Face-Local v1.2.3 megjott." in text
    assert "macOS es Windows" in text
    assert "https://github.com/example/face-local/releases/tag/v1.2.3" in text
    assert len(text) <= post_x_release.MAX_POST_LENGTH


def test_build_release_post_text_trims_variants_to_fit_limit() -> None:
    text = post_x_release.build_release_post_text(
        app_name="Face-Local",
        tag="v1.2.3",
        release_url="https://github.com/example/face-local/releases/tag/v1.2.3?with=very-long-query-string-to-force-a-shorter-variant",
        successful_platforms=["macOS", "Windows", "Linux"],
    )

    assert text.startswith("Face-Local v1.2.3")
    assert "https://github.com/example/face-local/releases/tag/v1.2.3" in text
    assert len(text) <= post_x_release.MAX_POST_LENGTH
