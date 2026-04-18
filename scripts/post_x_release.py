#!/usr/bin/env python3
"""Post a release announcement to X after a successful GitHub release."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

API_URL = "https://api.x.com/2/tweets"
MAX_POST_LENGTH = 280
REQUIRED_ENV_VARS = (
    "X_API_KEY",
    "X_API_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_TOKEN_SECRET",
)
DEFAULT_TEMPLATE = (
    "{app_name} {tag} megjott.\n\n"
    "Zero cloud. Zero upload. Minden helyben fut.\n"
    "Uj build-ek: {platforms}.\n\n"
    "Letoltes: {release_url}\n"
    "#FaceLocal #LocalAI #Privacy"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-name", default="Face-Local", help="Display name of the app.")
    parser.add_argument("--tag", required=True, help="Release tag, for example v1.2.3.")
    parser.add_argument("--release-url", required=True, help="Public URL of the GitHub release.")
    parser.add_argument(
        "--platform",
        action="append",
        default=[],
        help="Platform build result in NAME=RESULT format, for example macOS=success.",
    )
    parser.add_argument(
        "--template",
        default=os.environ.get("X_POST_TEMPLATE", ""),
        help=(
            "Optional custom template. Supported placeholders: "
            "{app_name}, {tag}, {version}, {platforms}, {release_url}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated post without sending it to X.",
    )
    args = parser.parse_args()

    successful_platforms = parse_successful_platforms(args.platform)
    post_text = build_release_post_text(
        app_name=args.app_name,
        tag=args.tag,
        release_url=args.release_url,
        successful_platforms=successful_platforms,
        template=args.template or None,
    )

    if args.dry_run:
        print(post_text)
        return 0

    credentials = load_credentials_from_env()
    if credentials is None:
        print(
            "Skipping X post: set X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, and "
            "X_ACCESS_TOKEN_SECRET to enable release announcements.",
            file=sys.stderr,
        )
        return 0

    tweet_id = create_post(text=post_text, credentials=credentials)
    print(f"Posted release announcement to X with id {tweet_id}.")
    return 0


def parse_successful_platforms(entries: Iterable[str]) -> list[str]:
    successful: list[str] = []
    for entry in entries:
        if "=" not in entry:
            successful.append(entry.strip())
            continue
        name, result = entry.split("=", 1)
        if result.strip().lower() == "success":
            successful.append(name.strip())
    return successful


def build_release_post_text(
    *,
    app_name: str,
    tag: str,
    release_url: str,
    successful_platforms: list[str],
    template: str | None = None,
) -> str:
    version = tag[1:] if tag.startswith("v") else tag
    platform_text = format_platform_list(successful_platforms) or "macOS, Windows es Linux"
    context = {
        "app_name": app_name,
        "tag": tag,
        "version": version,
        "platforms": platform_text,
        "release_url": release_url,
    }

    if template:
        text = template.format(**context).strip()
        ensure_post_length(text)
        return text

    variants = [
        DEFAULT_TEMPLATE.format(**context).strip(),
        DEFAULT_TEMPLATE.replace("\n#FaceLocal #LocalAI #Privacy", "").format(**context).strip(),
        (
            "{app_name} {tag} megjott.\n\n"
            "Zero cloud. Zero upload. Minden helyben fut.\n\n"
            "Letoltes: {release_url}"
        ).format(**context).strip(),
    ]
    for variant in variants:
        if len(variant) <= MAX_POST_LENGTH:
            return variant

    compact = "{app_name} {tag} live now: {release_url}".format(**context)
    ensure_post_length(compact)
    return compact


def format_platform_list(platforms: list[str]) -> str:
    cleaned = [platform.strip() for platform in platforms if platform.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} es {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])} es {cleaned[-1]}"


def ensure_post_length(text: str) -> None:
    if len(text) > MAX_POST_LENGTH:
        raise ValueError(f"Generated X post is too long ({len(text)} characters).")


def load_credentials_from_env() -> dict[str, str] | None:
    values = {name: os.environ.get(name, "").strip() for name in REQUIRED_ENV_VARS}
    populated = [name for name, value in values.items() if value]
    if not populated:
        return None

    missing = [name for name, value in values.items() if not value]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Missing required X credentials: {missing_list}")
    return values


def create_post(*, text: str, credentials: dict[str, str]) -> str:
    payload = json.dumps({"text": text}).encode("utf-8")
    authorization = build_oauth_header(
        method="POST",
        url=API_URL,
        consumer_key=credentials["X_API_KEY"],
        consumer_secret=credentials["X_API_SECRET"],
        token=credentials["X_ACCESS_TOKEN"],
        token_secret=credentials["X_ACCESS_TOKEN_SECRET"],
    )
    request = urllib.request.Request(
        url=API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "User-Agent": "face-local-release-bot",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Failed to post to X ({exc.code}): {body}") from exc

    data = json.loads(raw.decode("utf-8"))
    tweet_id = data.get("data", {}).get("id")
    if not tweet_id:
        raise SystemExit(f"X API did not return a tweet id: {data}")
    return str(tweet_id)


def build_oauth_header(
    *,
    method: str,
    url: str,
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
) -> str:
    oauth_params = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": token,
        "oauth_version": "1.0",
    }

    signature = build_oauth_signature(
        method=method,
        url=url,
        oauth_params=oauth_params,
        consumer_secret=consumer_secret,
        token_secret=token_secret,
    )
    oauth_params["oauth_signature"] = signature

    parts = [
        f'{percent_encode(key)}="{percent_encode(value)}"'
        for key, value in sorted(oauth_params.items())
    ]
    return "OAuth " + ", ".join(parts)


def build_oauth_signature(
    *,
    method: str,
    url: str,
    oauth_params: dict[str, str],
    consumer_secret: str,
    token_secret: str,
) -> str:
    normalized_url = normalize_url(url)
    query_params = urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True)
    signature_params = list(query_params) + list(oauth_params.items())
    parameter_string = "&".join(
        f"{percent_encode(key)}={percent_encode(value)}"
        for key, value in sorted((str(key), str(value)) for key, value in signature_params)
    )
    base_string = "&".join(
        [
            method.upper(),
            percent_encode(normalized_url),
            percent_encode(parameter_string),
        ]
    )
    signing_key = f"{percent_encode(consumer_secret)}&{percent_encode(token_secret)}"
    digest = hmac.new(signing_key.encode("utf-8"), base_string.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    return urllib.parse.urlunsplit((scheme, netloc, parsed.path, "", ""))


def percent_encode(value: str) -> str:
    return urllib.parse.quote(str(value), safe="-._~")


if __name__ == "__main__":
    raise SystemExit(main())
