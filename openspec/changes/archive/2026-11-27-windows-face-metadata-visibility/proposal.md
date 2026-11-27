# Proposal

## Why

Issue #111 ("EXIF adatok közé mentés") is broken on Windows in two verified ways. First, the person data does land in the file (EXIF `UserComment` with an `ASCII` prefix and UTF-8 JSON, or a custom `facelocal:data` XMP property), but Windows Explorer and the Photos app read neither, so Properties → Details shows no comment, no tags and no people. Users conclude that nothing was written. Second, the image browser's "save metadata" action rewrites each file three times in a row (GPS, date, persons). Each rewrite makes antivirus or the Search indexer open the fresh file. A reader that holds the file with share-read only (Photos app, many viewers, AV) defeats both `os.replace` and the in-place fallback, and the data silently ends up in a sidecar JSON.

## What Changes

- Embedded face metadata for JPEG also goes into fields that Windows and common photo tools display:
  - EXIF `XPComment` and `XPKeywords` (UTF-16LE): person names → Explorer "Comments" / "Tags".
  - XMP Microsoft Photo `MP:RegionInfo`: named face rectangles → Explorer/Photos "People".
  - XMP MWG `mwg-rs:Regions`: named face rectangles for Lightroom, digiKam and similar tools.
- The `facelocal.faces.v1` JSON stays embedded in **both** EXIF `UserComment` and XMP (`facelocal:data`), with identical content, and remains the round-trip source the app reads back. Existing files written by the current version stay readable.
- Non-ASCII text is never written to ASCII-only EXIF fields (`ImageDescription`, `UserComment` with the `ASCII` code). JSON in `UserComment` uses a standards-correct character code.
- GPS, date and person metadata chosen in one "save metadata" action are written to each file in **one** rewrite instead of three.
- When the target file is locked, the result is reported as **locked** (not as a silent sidecar success). The data is still preserved in a sidecar, and the summary names the locked files so the user can close the other program and retry.
- Rewrites keep the ICC color profile and other existing metadata (EXIF incl. GPS/date/orientation, unrelated XMP).
- PNG keeps its current XMP-based behavior, extended with the same region/description XMP.

Out of scope: HEIC/RAW embedding (still sidecar), Windows Photos "People" recognition integration beyond writing the standard fields, lossless JPEG segment surgery (see design for the re-encode decision), background export.

## Capabilities

### New Capabilities
- `face-metadata-embedding`: how person/face data is written into and read back from image files (fields used, visibility in OS tools, preservation of existing metadata, single-write behavior, lock handling and result reporting).

### Modified Capabilities
<!-- none: no existing specs in openspec/specs/ -->

## Impact

- `app/utils/image_metadata.py`: field writers (XP tags, MP/MWG XMP), UserComment encoding, ICC preservation, lock detection/result mode.
- `app/utils/exif.py`: GPS/date writers become composable with the face write (one rewrite).
- `app/services/face_metadata_export_service.py`: region payload (normalized, orientation-aware boxes), new "locked" outcome in results/summary.
- `app/ui/panels/image_browser/image_loading.py`: single combined write per image; summary shows locked files.
- `app/ui/dialogs/export_dialog.py`, `app/workers/face_metadata_export_worker.py`: summary shows locked count.
- i18n strings (EN + HU) for the locked outcome.
- Tests: `tests/test_face_metadata_export_service.py`, `tests/test_exif_write.py`, new Windows-field tests.
- No new mandatory dependency (Pillow path must work without `piexif`). No DB schema change.
