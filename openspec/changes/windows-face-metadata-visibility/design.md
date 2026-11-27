# Design

## Context

See proposal.md (Why) and `specs/face-metadata-embedding/spec.md` for the required behavior.

Current state, verified on Windows 10 with Pillow 11.3 and **no `piexif` installed**:

- `app/utils/image_metadata.py` writes the `facelocal.faces.v1` JSON either to EXIF `UserComment` (`ASCII\0\0\0` + UTF-8 bytes, default for JPEG) or to a custom XMP property `facelocal:data`. Explorer shows neither. Merging XMP is regex-based (`_merge_xmp`).
- Every write decodes the image with Pillow and **re-encodes** it (`quality="keep"`) to a temp file, then `_replace_atomic` swaps it in. `icc_profile` is not passed, so the color profile is dropped. Other APP segments (MPF, Photoshop IRB, etc.) are dropped too.
- `_replace_atomic` retries `os.replace` 6 times (~1.7 s total), then overwrites the original **in place** (`r+b`). That is not atomic: a crash mid-write can corrupt the original.
- `app/utils/exif.py` `write_exif_gps` / `write_exif_date` each do their own full rewrite through the same path.
- `ImageBrowserPanel` "save metadata" (`image_loading.py:358-387`) calls GPS write, date write, then `FaceMetadataExportService.export_images`: three rewrites per file within milliseconds.
- Face boxes are stored in the EXIF-oriented (upright) pixel space; the payload already records `coordinate_space: "exif_oriented"` and `exif_orientation`.

Experiment results that shaped this design:

| File content | Explorer Comment | Title | Keywords | PeopleNames |
|---|---|---|---|---|
| Current output (UserComment JSON) | – | – | – | – |
| XPComment + XPKeywords (UTF-16LE) + ImageDescription | yes | `G?za` | yes | – |
| XMP `MP:RegionInfo` + `dc:description` | – | yes | – | yes |

A file held with `FileShare.Read` by another process makes both `os.replace` and the in-place fallback raise `PermissionError` → sidecar.

## Goals / Non-Goals

**Goals:**
- JPEG metadata edits become lossless byte splices; pixel data is never re-encoded.
- One composable "metadata edit" applied in a single file replace, shared by the GPS, date and face writers.
- A distinct `locked` outcome that is visible all the way to the UI summary.

**Non-Goals:**
- Extended XMP (multi-segment XMP > 64 KB).
- Reading other tools' regions back into the app as face assignments.
- A background retry queue. Retry is the user re-running the export.

## Decisions

### D1. JPEG writes splice APP segments instead of re-encoding
A small JPEG segment splicer (in `app/utils/`, stdlib only) walks the markers from SOI to SOS, replaces or inserts the APP1 `Exif` segment and the APP1 XMP (`http://ns.adobe.com/xap/1.0/`) segment, and copies every other byte verbatim (APP0/JFIF, APP2 ICC, MPF, DQT, scan data).
- EXIF bytes are still produced by `PIL.Image.Exif.tobytes()` (piexif optional, as today).
- This makes ICC/MPF preservation automatic and removes generation loss.
- *Alternatives:* keep Pillow re-encode and only pass `icc_profile` (still lossy, still drops MPF and other APP segments); `piexif.insert` (EXIF only, no XMP, optional dependency); exiftool (external binary, not bundled).
- PNG keeps the Pillow re-save (lossless), now passing `icc_profile` and `exif` through.

### D2. Field layout
| Data | Field | Encoding |
|---|---|---|
| Full `facelocal.faces.v1` JSON | EXIF `UserComment` (0x9286) | `UNICODE\0` code + UTF-16 in the EXIF block's byte order |
| Full `facelocal.faces.v1` JSON (same document) | XMP `facelocal:data` (own `rdf:Description`) | UTF-8, XML-escaped |
| Known person names | EXIF 0th `XPComment` (0x9C9C), joined with `; ` | UTF-16LE + NUL |
| Known person names | EXIF 0th `XPKeywords` (0x9C9E), `;`-separated | UTF-16LE + NUL |
| Named face rectangles | XMP `MP:RegionInfo/MPRI:Regions` (`MPReg:Rectangle`, `MPReg:PersonDisplayName`) | normalized `x, y, w, h` (top-left) |
| Named face rectangles | XMP `mwg-rs:Regions` (`mwg-rs:Type=Face`, `stArea:unit=normalized`, center `x,y` + `w,h`, `AppliedToDimensions`) | normalized |

- For JPEG the JSON is written to **both** `UserComment` and XMP, in one write and from the same serialized document (decision by the project owner). `UserComment` keeps the data visible among standard EXIF fields (exiftool, EXIF viewers); XMP is what MWG/XMP tools read. PNG has no EXIF and keeps XMP only.
- `UserComment` switches from `ASCII\0\0\0` + UTF-8 (mismatched code, mojibake for accents) to `UNICODE\0` + UTF-16 in the byte order of the EXIF block (Pillow writes little-endian `II`). This matches the EXIF spec, so accented names read correctly.
  - *Alternative considered:* keep the ASCII code but serialize with `ensure_ascii=True` (`é` escapes). Half the size, but names are unreadable to humans in EXIF viewers. Used only as the size fallback (see Risks).
- Each export overwrites our `UserComment` (it is owned by Face-Local once it holds a `facelocal.faces.v1` document). A foreign `UserComment` (not parseable as our schema) is **not** overwritten: the JSON then goes to XMP only and `error_message` says so.
- Reading prefers XMP, then `UserComment`. If both exist and differ (edited by another tool), XMP wins and the next export rewrites both.
- `ImageDescription` is never written (ASCII-only per spec, shows `?` in Explorer).
- `FaceMetadataExportOptions.embed_in_exif_comment` keeps its meaning: True (default) writes the `UserComment` copy, False writes XMP only. The success write mode for JPEG becomes `xmp` when only XMP was written and stays `exif_user_comment` when both were written, so existing UI counts keep working.

### D3. Ownership of tags and regions
Face-Local owns `XPComment`, `XPKeywords`, `MP:RegionInfo` and `mwg-rs:Regions` on files it exports. Each export replaces these four entirely with the current list (so removed persons disappear, satisfying "replaces, never duplicates").
- Rationale: the app has no reliable way to merge another tool's regions with its own, and partial merges caused duplicates in the issue discussion.
- The trade-off is documented in the export warning text: names/tags set by other tools in these fields are overwritten.
- When the export has no known names (names excluded or only unknown faces), `XPComment`/`XPKeywords` are removed rather than left stale. The same applies to regions when boxes are excluded.

### D4. XMP editing with ElementTree
Replace the regex `_merge_xmp` with `xml.etree.ElementTree` editing of the RDF tree:
- remove our owned properties
- add ours
- keep every other `rdf:Description` child
- register the namespace prefixes so foreign prefixes round-trip

If the existing XMP cannot be parsed, the image is not modified: the result is `failed` with "existing XMP unreadable", plus a sidecar. Silently dropping foreign XMP would violate "Existing metadata preserved".
- *Alternative:* keep regex merging. Rejected: it cannot safely remove nested `MP:RegionInfo` / `mwg-rs:Regions` structures.

### D5. Region coordinates
Boxes are in the EXIF-oriented (upright) pixel space.
- Normalize against the **oriented** dimensions (raw width/height swapped for orientations 5–8).
- Write `mwg-rs:AppliedToDimensions` with those oriented dimensions.
- MP uses top-left `x, y, w, h`; MWG uses center `x, y` and `w, h`.
- Clamp to [0, 1] and round to 6 decimals.

Both Windows Photo Gallery and digiKam treat these regions as relative to the displayed (oriented) image. This must be confirmed with a manual check on an orientation-6 sample (task 6.3). If the check fails, only the normalization function changes.

### D6. Single write per action
Introduce one metadata-edit API in `image_metadata`, for example `apply_image_metadata(path, *, gps=None, date=None, faces_payload=None, …) -> MetadataWriteResult`:
1. read the current EXIF/XMP once
2. apply all edits to the in-memory EXIF and XMP
3. splice once
4. replace once

- `write_exif_gps` / `write_exif_date` become thin wrappers around it. Their signatures and bool return stay as they are, so the other callers (`metadata.py`, `image_metadata_dialog.py`) are unchanged.
- `FaceMetadataExportService.export_images` gets an optional per-image extra-edits hook (`extra_edits: Callable[[Image], dict] | None`). The image browser passes GPS/date through it, which removes its separate GPS/date loop.

### D7. Replace and lock handling
- Write the temp file next to the image (same volume), `fsync`, then `os.replace` with retry.
- Retry budget raised to about 3 s with backoff, which covers indexer/AV bursts.
- The **in-place `r+b` fallback is removed** (non-atomic, can corrupt the original).
- If the replace still fails with a sharing/permission error (`PermissionError`, WinError 5/32/33), the temp file is deleted, the sidecar is written, and the result is the new `WRITE_MODE_LOCKED = "locked"` with the error message.
- `MetadataWriteResult.ok` stays True for `locked`, because the data is preserved.
- `FaceMetadataExportSummary` gets `locked_count` and `locked_paths`. UI summaries (image browser, export dialog, worker) show them with an EN+HU string telling the user to close the program holding the file and retry.
- A genuinely read-only file (`os.access` false) keeps today's sidecar behavior with its reason.

### D8. App's own file handles
The image loaders already use context managers or Qt readers that close after decoding. One check is needed: `image_utils.py:243`, `PilImage.open(path).convert("RGB")` without `with`. Pillow closes single-frame files after `load()`, but it is made explicit with `with` so the app is never the lock holder.

## Risks / Trade-offs

- [Malformed JPEG (missing SOI, truncated segments)] → the splicer validates structure and raises. The result is `failed` plus a sidecar, and the original is untouched.
- [EXIF APP1 > 64 KB after adding UserComment JSON (UTF-16 doubles size) and XP tags] → step down: first rewrite `UserComment` as ASCII-code JSON with `ensure_ascii=True`; if still too large, omit the `UserComment` copy; then drop XP tags. Each step is reported in `error_message`. The JSON and regions always still go to XMP.
- [XMP packet > 64 KB with hundreds of faces] → fall back to sidecar for the whole payload (mode `sidecar`, reason "XMP too large").
- [Overwriting other tools' person tags/regions (D3)] → explicit in the warning dialog text, and the user can choose sidecar-only.
- [Windows-only behavior cannot be fully tested in CI on Linux/macOS] → byte-level unit tests for all fields, plus a Windows-only test that reads Shell properties, skipped elsewhere, and a manual checklist task.
- [Lock still persists after retry] → reported as `locked`, data in sidecar; user retries.

## Migration Plan

- No DB change.
- Files exported by earlier versions stay readable (UserComment ASCII or `facelocal:data`). On the next export the ASCII-coded UserComment is rewritten with the `UNICODE` code and the same document is added to XMP.
- Rollback: the previous release checks the `facelocal:data` XMP first, so files written by this version remain readable by it. (It would mis-decode the new UTF-16 `UserComment` on its own, but never reaches it while the XMP copy exists.)
