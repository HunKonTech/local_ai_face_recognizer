# Tasks

## 1. Lossless JPEG segment splicer (design D1)

- [x] 1.1 Add a stdlib-only JPEG APP-segment splicer in `app/utils/` (new module, e.g. `jpeg_segments.py`) that reads EXIF/XMP APP1 payloads and writes a copy with replaced/inserted/removed EXIF and XMP segments, copying all other bytes verbatim; verify with new `tests/test_jpeg_segments.py`: scan data and APP2 ICC bytes are identical before/after, round-trip of EXIF and XMP payloads, malformed input (no SOI, truncated segment) raises a clear error.
- [x] 1.2 Enforce the 65 533-byte APP1 limit in the splicer and raise a dedicated "segment too large" error; verify with a unit test feeding an oversize XMP payload.

## 2. Safe replace and locked outcome (design D7, D8)

- [x] 2.1 Rework `_replace_atomic` in `app/utils/image_metadata.py`: fsync temp file, ~3 s backoff retry, remove the in-place `r+b` fallback, always delete the temp file on failure, raise a typed `FileLockedError` for sharing/permission errors; verify with a test that monkeypatches `os.replace` to raise `PermissionError` and asserts the original is byte-identical and no `*.facelocal.tmp` remains.
- [x] 2.2 Add `WRITE_MODE_LOCKED`; on `FileLockedError` write the sidecar and return `locked` with the error message (`ok` stays True); verify with a unit test, plus a Windows-only test (`skipif not win32`) that holds the file via `FileShare.Read` (e.g. `msvcrt`/`ctypes CreateFileW` share-read) and expects `locked` + sidecar.
- [x] 2.3 Make the `PilImage.open(path).convert("RGB")` in `app/utils/image_utils.py` (~line 243) use a context manager; verify `ruff check app/utils` and existing image-utils tests pass.

## 3. Field writers and XMP editing (design D2, D3, D4)

- [x] 3.1 Replace regex `_merge_xmp` with an ElementTree-based XMP editor that removes/sets the owned properties (`facelocal:data`, `MP:RegionInfo`, `mwg-rs:Regions`) and keeps all foreign properties and prefixes; unparsable existing XMP raises; verify with tests: foreign `xmp:Rating` survives, repeated edits leave one copy of each owned property, unparsable XMP → error.
- [x] 3.2 Build `XPComment` / `XPKeywords` (UTF-16LE + NUL) from known person names via `PIL.Image.Exif`, removing them when there are no names; stop writing `ImageDescription`; verify with tests decoding the tags back to "Szilvay Géza" and no `piexif` required (test with `piexif` import blocked).
- [x] 3.3 Build MP and MWG region XML from the payload (design D5: normalized to oriented dimensions, MP top-left, MWG center, `AppliedToDimensions`, clamp, 6 decimals); verify with parametrized tests for orientations 1, 3, 6, 8 against expected normalized values.
- [x] 3.4 Write the same JSON document to `UserComment` (`UNICODE\0` + UTF-16 in the EXIF byte order) alongside XMP when `embed_in_exif_comment` is True; overwrite only an empty or Face-Local-owned `UserComment`; apply the size step-down (ASCII-escaped JSON, then omit) from design Risks; verify with tests: UserComment and XMP decode to equal documents with "Szilvay Géza" readable, a legacy ASCII UserComment is rewritten as UNICODE, a foreign UserComment is left untouched with a reason in `error_message`, an oversize payload triggers the fallback and is reported, and `embed_in_exif_comment=False` writes XMP only.
- [x] 3.5 Keep reading legacy data: `read_face_metadata` still recovers JSON from ASCII `UserComment`, `UNICODE` (UTF-16) `UserComment`, and `facelocal:data`; verify with fixtures written in the old format.

## 4. Single-write metadata API (design D6)

- [x] 4.1 Add `apply_image_metadata(path, *, gps=None, date=None, faces_payload=None, prefer_sidecar_only=False)` in `app/utils/image_metadata.py` that reads once, applies all edits, splices once (JPEG) or re-saves once with `icc_profile`/`exif` preserved (PNG), and returns `MetadataWriteResult`; route `write_face_metadata` through it; verify with tests: GPS + DateTimeOriginal + orientation + ICC unchanged after a face export, and a spy on the replace function shows exactly one replace for a combined GPS+date+faces call.
- [x] 4.2 Turn `write_exif_gps` / `write_exif_date` in `app/utils/exif.py` into wrappers over `apply_image_metadata` (same signatures, bool return); verify `tests/test_exif_write.py` passes unchanged and JPEG pixel bytes stay identical after a GPS write.
- [x] 4.3 Update the `FaceMetadataExportOptions.embed_in_exif_comment` docstring (True = UserComment copy in addition to XMP, False = XMP only) and add the optional `extra_edits` hook to `FaceMetadataExportService.export_images`; verify with `tests/test_face_metadata_export_service.py` (existing tests updated to the new write modes; new test passing GPS/date via `extra_edits`).

## 5. Reporting and UI

- [x] 5.1 Add `locked_count` / `locked_paths` to `FaceMetadataExportSummary` and exclude locked results from `sidecar_count`; verify with a service test mixing embedded, sidecar, locked and failed results.
- [x] 5.2 Switch the image browser "save metadata" action (`app/ui/panels/image_browser/image_loading.py` ~358-400) to a single `export_images(..., extra_edits=...)` call and show locked files in its summary; verify with `QT_QPA_PLATFORM=offscreen` test or manual run that GPS/date/person counts are still reported.
- [x] 5.3 Show the locked count/list in `app/ui/dialogs/export_dialog.py` and `app/workers/face_metadata_export_worker.py` summaries; verify `tests/test_face_metadata_export_worker.py` passes and covers the locked field.
- [x] 5.4 Add EN+HU i18n strings for the locked message and update the export warning text to say that person comments/tags/regions written by other tools are overwritten (design D3); verify with `grep` that every new key has both `en` and `hu`, and `ruff check app/ui`.

## 6. Windows integration check

- [x] 6.1 Add a Windows-only test (`skipif not win32`) that exports a sample JPEG and reads Shell properties (`System.Comment`, `System.Keywords`, `System.Photo.PeopleNames`) via PowerShell/COM, expecting the accented names; verify it passes on Windows and is skipped elsewhere.
- [x] 6.2 Run the full test suite (`QT_QPA_PLATFORM=offscreen python -m pytest -q`) and `ruff check app/`; verify no new failures beyond the known `tests/test_gdrive_*` ones.
- [x] 6.3 Manual check on Windows: export an orientation-6 phone photo, open it in Windows Photos / Explorer and digiKam (or another MWG reader), confirm names appear and regions sit on the right faces; record the result in the PR description (adjust only the normalization function if regions are misplaced).
- [x] 6.4 Manual check: hold a JPEG open with FileShare.Read (another program, simulated from PowerShell; the modern Photos app does not keep the file open), export it, confirm the summary reports it as locked with a sidecar; release it, export again, confirm it embeds.
