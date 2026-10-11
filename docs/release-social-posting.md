# Release social posting (Buffer)

This repository has release-to-social automation via Buffer.

Current implementation:
- Workflow file: `.github/workflows/build-release.yml`
- Posting script: `scripts/post_buffer_release.py`
- Tests: `tests/test_post_buffer_release.py`

Behavior:
- After the `release` job succeeds, a separate `post-to-buffer` job runs.
- It runs only for `push` events and only on the first workflow attempt:
  `needs.release.result == 'success' && github.event_name == 'push' && github.run_attempt == 1`
- The posting step uses `continue-on-error: true`, so social posting must never fail the release.
- Default posting mode is `shareNow`, so the post should publish immediately.
- If `BUFFER_POST_MODE` is empty, the script falls back to `shareNow`.

Required org/repo secret:
- `BUFFER_API_KEY`

Optional org/repo secrets:
- `BUFFER_CHANNEL_ID` — recommended if multiple Buffer channels exist
- `BUFFER_CHANNEL_NAME` — optional fallback selector when channel ID is not set
- `BUFFER_ORGANIZATION_ID` — optional, otherwise the script auto-discovers organizations
- `BUFFER_CHANNEL_SERVICE` — optional, defaults to `twitter`
- `BUFFER_POST_MODE` — optional, defaults to `shareNow`
- `BUFFER_POST_TEMPLATE` — optional custom post template

Buffer posting details:
- The script creates a text post through Buffer's GraphQL API at `https://api.buffer.com`
- It auto-selects the first connected `twitter` channel when no explicit channel secret is set
- If multiple matching channels exist, the script picks the first sorted match and prints a warning
- The template supports:
  `{app_name}`, `{tag}`, `{version}`, `{platforms}`, `{release_url}`
- Successful platform names are derived from the build job results
- The generated text is capped at 280 characters

Recommended setup for this repo:
- Set `BUFFER_API_KEY`
- Leave `BUFFER_POST_MODE` empty or set it explicitly to `shareNow`
- If more than one Buffer/X account is connected, set `BUFFER_CHANNEL_ID`

Useful local checks:

```bash
pytest -q tests/test_post_buffer_release.py

BUFFER_POST_MODE='' python3 scripts/post_buffer_release.py \
  --dry-run \
  --app-name "Face-Local" \
  --tag "v1.2.3" \
  --release-url "https://github.com/example/repo/releases/tag/v1.2.3" \
  --platform "macOS=success" \
  --platform "Windows=success"
```
