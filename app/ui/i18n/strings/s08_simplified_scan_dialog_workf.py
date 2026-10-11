"""UI strings (EN + HU) — part 8, starting at: Simplified scan dialog workflows."""

from __future__ import annotations

from typing import Dict

STRINGS: Dict[str, Dict[str, str]] = {
    # ── Simplified scan dialog workflows ──────────────────────────────────────
    "workflow_face_detection_title":
        {"en": "Face Detection & Assignment",
         "hu": "Arcfelismerés és hozzárendelés"},
    "workflow_face_detection_desc":
        {"en": "Indexes new images, detects faces, embeds, and assigns known persons.",
         "hu": "Új képeket indexel, arcokat detektál és emberekhez rendeli."},
    "workflow_train_model_title":
        {"en": "Train AI Model",
         "hu": "AI modell tanítása"},
    "workflow_train_model_desc":
        {"en": "Retrains the deep classifier on all labeled persons.",
         "hu": "Újratanítja az AI modellt az összes elnevezett emberen."},
    "workflow_full_rescan_title":
        {"en": "Full Rescan",
         "hu": "Teljes újrabeolvasás"},
    "workflow_full_rescan_desc":
        {"en": "Clears all detections and rebuilds embeddings and clusters from scratch.",
         "hu": "Felszabadítja az összes detektálást és újraépíti az arcokat."},
    "button_start":
        {"en": "Start",
         "hu": "Kezdés"},

    "retro_verify_title":
        {"en": "Re-verify Stored Faces (multi-technology)",
         "hu": "Tárolt arcok újra-ellenőrzése (több technológia)"},
    "retro_verify_unavailable":
        {"en": "The crop-level face verifier is unavailable (model missing or "
               "verification disabled in config). Enable verification_enabled in "
               "your config and ensure the YuNet model is present.",
         "hu": "A vágat-szintű arc-verifikátor nem elérhető (a modell hiányzik vagy "
               "a verification_enabled ki van kapcsolva a konfigban). Kapcsold be a "
               "verification_enabled opciót, és ellenőrizd, hogy a YuNet modell megvan."},
    "retro_verify_confirm":
        {"en": "Checked {scanned} face(s) across {images} image(s).\n\n"
               "{n} face(s) could not be re-confirmed as real faces and can be removed "
               "(unassigned or automatically assigned only).\n"
               "{flagged} manually assigned / drawn face(s) also failed — kept for "
               "your review and NOT deleted.\n"
               "{missing} image file(s) could not be loaded and were skipped.\n\n"
               "Delete the {n} unconfirmed non-face(s) now? This cannot be undone.",
         "hu": "{images} képen {scanned} arcot ellenőrzött.\n\n"
               "{n} arc nem erősíthető meg valódi arcként, és eltávolítható "
               "(csak hozzárendeletlen vagy automatikusan hozzárendelt).\n"
               "{flagged} kézzel hozzárendelt / rajzolt arc szintén bukott — "
               "felülvizsgálatra megmarad, NEM töröljük.\n"
               "{missing} képfájl nem volt betölthető, azokat kihagytuk.\n\n"
               "Törölhető most a(z) {n} nem megerősített nem arc? Ez nem vonható vissza."},
    "retro_verify_none":
        {"en": "Checked {scanned} face(s) across {images} image(s) — "
               "all confirmed as real faces.\n"
               "({flagged} manually assigned face(s) failed but are kept for review.)\n"
               "({missing} image file(s) skipped — files missing from disk.)",
         "hu": "{images} képen {scanned} arcot ellenőrzött — "
               "minden megerősítve valódi arcként.\n"
               "({flagged} kézzel hozzárendelt arc bukott, de felülvizsgálatra megmarad.)\n"
               "({missing} képfájl kihagyva — fájlok hiányoznak a lemezről.)"},
    "retro_verify_status":
        {"en": "Removed {deleted} unconfirmed non-face(s)",
         "hu": "{deleted} meg nem erősített nem arc eltávolítva"},

    # Identity repair results dialog
    "repair_title":      {"en": "Identity Repair — Merge Fragmented Unknowns",
                          "hu": "Identitás-helyreállítás — Töredékek összevonása"},
    "repair_intro":      {"en": "Pairs of 'Unknown' identities that appear to be the "
                                "same person. Keep the pairs you want to merge, then "
                                "confirm. Transitively linked pairs are consolidated "
                                "into one identity.",
                          "hu": "Olyan „Unknown” identitáspárok, amelyek azonos "
                                "személynek tűnnek. Hagyd bejelölve az összevonandó "
                                "párokat, majd erősítsd meg. A láncban összekapcsolt "
                                "párok egyetlen identitássá olvadnak."},
    "repair_count":      {"en": "{n} merge candidate(s)", "hu": "{n} összevonási javaslat"},
    "repair_no_matches_title": {"en": "No fragments found", "hu": "Nincs töredék"},
    "repair_no_matches_msg":   {"en": "No fragmented Unknown identities were found.",
                                "hu": "Nem található széttöredezett „Unknown” identitás."},
    "repair_pair_label": {"en": "{a} ≈ {b}  ({pct}% match, {fa}+{fb} faces)",
                          "hu": "{a} ≈ {b}  ({pct}% egyezés, {fa}+{fb} arc)"},
    "repair_merge_btn":  {"en": "Merge Selected", "hu": "Kijelöltek összevonása"},
    "repair_confirm_title": {"en": "Confirm merge", "hu": "Összevonás megerősítése"},
    "repair_confirm_msg":   {"en": "Merge {n} candidate pair(s)? This consolidates the "
                                   "linked Unknown identities and cannot be auto-undone.",
                             "hu": "Összevonod a(z) {n} javasolt párt? Ez egyesíti a "
                                   "kapcsolódó „Unknown” identitásokat, és nem vonható "
                                   "vissza automatikusan."},
    "repair_done_status":   {"en": "Identity repair: {groups} group(s) consolidated, "
                                   "{merged} fragment(s) merged away",
                             "hu": "Identitás-helyreállítás: {groups} csoport egyesítve, "
                                   "{merged} töredék összevonva"},
    "repair_error":      {"en": "Identity repair failed: {error}",
                          "hu": "Az identitás-helyreállítás sikertelen: {error}"},

    # ── Face diagnostics ("why this identity?") ───────────────────────────
    "diag_menu":         {"en": "Why this identity?", "hu": "Miért ez az identitás?"},
    "diag_title":        {"en": "Face Diagnostics — face #{id}",
                          "hu": "Arc diagnosztika — arc #{id}"},
    "diag_current":      {"en": "Current identity", "hu": "Jelenlegi identitás"},
    "diag_source":       {"en": "Assigned via", "hu": "Hozzárendelés módja"},
    "diag_threshold":    {"en": "Adaptive threshold", "hu": "Adaptív küszöb"},
    "diag_quality":      {"en": "Quality", "hu": "Minőség"},
    "diag_named_header": {"en": "Top named-person matches", "hu": "Legjobb nevesített egyezések"},
    "diag_unknown_header": {"en": "Top Unknown matches", "hu": "Legjobb „Unknown” egyezések"},
    "diag_verdict":      {"en": "Verdict", "hu": "Magyarázat"},
    "diag_none":         {"en": "(none)", "hu": "(nincs)"},
    "diag_error":        {"en": "Diagnostics failed: {error}",
                          "hu": "A diagnosztika sikertelen: {error}"},

    # ── Name suggestions ──────────────────────────────────────────────────
    "suggestions_btn":   {"en": "Name Suggestions", "hu": "Névajánlatok"},
    "suggestions_title": {"en": "Name Suggestions — Unknown → Known",
                          "hu": "Névajánlatok — Ismeretlen → Ismert"},
    "suggestions_similarity": {"en": "{pct}% match", "hu": "{pct}% egyezés"},
    "suggestions_gallery_title":    {"en": "{name} — All Faces", "hu": "{name} — Összes arc"},
    "suggestions_compare_title":    {"en": "Compare: {cand} vs {target}",
                                     "hu": "Összehasonlítás: {cand} vs {target}"},
    "suggestions_full_img_title":   {"en": "Full Image", "hu": "Teljes kép"},
    "suggestions_candidate":        {"en": "Unknown", "hu": "Ismeretlen"},
    "suggestions_target":           {"en": "Named person", "hu": "Ismert személy"},
    "suggestions_no_faces":         {"en": "No faces found.", "hu": "Nincs arc."},
    "gallery_exclude_from_merge":   {"en": "Exclude from merge",
                                     "hu": "Kihagyás az összevonásból"},
    "gallery_include_in_merge":     {"en": "Undo exclusion",
                                     "hu": "Kizárás visszavonása"},
    "gallery_excluded_tooltip":     {"en": "Excluded from merge — will stay separate",
                                     "hu": "Kizárva az összevonásból — külön marad"},
    "gallery_merge_exclusion_hint": {"en": "Right-click a face to exclude it from the merge",
                                     "hu": "Jobb klikk egy arcra a kizáráshoz"},
    "gallery_excluded_count":       {"en": "{n} excluded from merge",
                                     "hu": "{n} kizárva az összevonásból"},
    "merge_all_excluded":           {"en": "All faces are excluded from the merge — "
                                           "nothing to merge.",
                                     "hu": "Minden arc ki van zárva az összevonásból — "
                                           "nincs mit összevonni."},
    "suggestions_click_full_image": {"en": "Click to open full image",
                                     "hu": "Kattints a teljes kép megnyitásához"},

    # ── Google Drive cache (settings read-only display) ────────────────────
    "gdrive_cache_group":      {"en": "Google Drive Cache",
                                "hu": "Google Drive cache"},
    "gdrive_cache_dir_label":  {"en": "Cache directory (read-only):",
                                "hu": "Cache mappa (csak megtekintés):"},
    "gdrive_cache_size_label": {"en": "Current size:",
                                "hu": "Jelenlegi méret:"},
    "gdrive_cache_size_value": {"en": "{mb:.1f} MB",
                                "hu": "{mb:.1f} MB"},
    "gdrive_cache_note":       {"en": "Temporary files downloaded from Google Drive. "
                                      "Cleaned up on every start and exit. "
                                      "Max 500 MB; older files are evicted automatically.",
                                "hu": "A Google Drive-ról letöltött ideiglenes fájlok. "
                                      "Minden indításkor és kilépéskor törlődnek. "
                                      "Maximum 500 MB; a régebbi fájlok automatikusan törlődnek."},
    "gdrive_offline_error":    {"en": "Google Drive mode requires an internet connection.",
                                "hu": "Google Drive módhoz internetkapcsolat szükséges."},

    # ── Google Drive mode (Settings tab + workflow) ────────────────────────
    "settings_tab_gdrive":    {"en": "Google Drive", "hu": "Google Drive"},
    "settings_tab_recording": {"en": "Recording", "hu": "Rögzítés"},
    "rec_set_output_dir":     {"en": "Output folder:", "hu": "Mentési mappa:"},
    "rec_set_browse":         {"en": "Browse…", "hu": "Tallózás…"},
    "rec_set_quality":        {"en": "Quality:", "hu": "Minőség:"},
    "rec_set_fps":            {"en": "Frame rate (FPS):", "hu": "Képkocka/mp (FPS):"},
    "rec_set_segment":        {"en": "Segment length (s):",
                               "hu": "Szegmens hossza (mp):"},
    "rec_set_cursor":         {"en": "Capture mouse cursor",
                               "hu": "Egérkurzor rögzítése"},
    "rec_set_microphone":     {"en": "Capture microphone",
                               "hu": "Mikrofon rögzítése"},
    "rec_set_system_audio":   {"en": "Capture system audio (if available)",
                               "hu": "Rendszerhang rögzítése (ha elérhető)"},
    "rec_set_concat":         {"en": "Merge segments into one file when stopping",
                               "hu": "Szegmensek összefűzése egy fájlba leállításkor"},
    "rec_set_ffmpeg_path":    {"en": "ffmpeg path (optional):",
                               "hu": "ffmpeg útvonal (opcionális):"},
    "rec_set_unset":          {"en": "(ask each time)", "hu": "(kérdezzen rá mindig)"},
    "rec_set_display_group":  {"en": "Screens to record", "hu": "Képernyők rögzítése"},
    "rec_set_mode_active":    {"en": "Active application window",
                               "hu": "Aktív alkalmazásablak"},
    "rec_set_mode_all":       {"en": "All screens", "hu": "Összes képernyő"},
    "rec_set_mode_selected":  {"en": "Selected screens", "hu": "Kiválasztott képernyők"},
    "rec_set_primary_marker": {"en": "(primary)", "hu": "(elsődleges)"},
    "rec_set_monitor_word":   {"en": "monitor", "hu": "monitor"},
    "rec_set_no_displays":    {"en": "(no monitors detected)",
                               "hu": "(nem észlelhető monitor)"},
    "rec_set_auto_fps":       {"en": "Reduce frame rate for multi-monitor capture",
                               "hu": "Képkockaszám csökkentése többmonitoros rögzítésnél"},
    "rec_set_backend":        {"en": "Windows capture mode:",
                               "hu": "Windows rögzítési mód:"},
    "rec_set_backend_auto":   {"en": "Automatic (recommended)",
                               "hu": "Automatikus (ajánlott)"},
    "rec_set_backend_ddagrab": {"en": "Desktop Duplication (ddagrab)",
                                "hu": "Asztalmásolás (ddagrab)"},
    "rec_set_backend_gdigrab": {"en": "Legacy GDI (gdigrab)",
                                "hu": "Régi GDI (gdigrab)"},
    "rec_set_backend_tip": {
        "en": "GDI capture can record an all-black video on hybrid-GPU, HDR or "
              "hardware-accelerated displays. Desktop Duplication records the "
              "composited desktop instead. Automatic tests both and keeps the "
              "one that produces a picture.",
        "hu": "A GDI rögzítés teljesen fekete videót adhat hibrid GPU-s, HDR "
              "vagy hardveresen gyorsított kijelzőkön. Az asztalmásolás a "
              "kompozitált asztalt rögzíti. Az automatikus mód mindkettőt "
              "leteszteli, és azt tartja meg, amelyik képet ad.",
    },
    "rec_set_preflight":      {"en": "Check for a black screen before recording",
                               "hu": "Fekete képernyő ellenőrzése rögzítés előtt"},
    "rec_set_test_capture":   {"en": "Test capture", "hu": "Rögzítés tesztelése"},
    "rec_test_running":       {"en": "Testing…", "hu": "Tesztelés…"},
    "rec_test_ok":            {"en": "Capture OK ({backend})",
                               "hu": "A rögzítés rendben ({backend})"},
    "rec_test_black":         {"en": "The captured image is BLACK ({backend})",
                               "hu": "A rögzített kép FEKETE ({backend})"},
    "rec_test_failed":        {"en": "Test failed: {error}",
                               "hu": "A teszt sikertelen: {error}"},
    "rec_backend_switched": {
        "en": "Switched to {backend} — the other capture mode produced a black "
              "image.",
        "hu": "Átváltás erre: {backend} — a másik rögzítési mód fekete képet "
              "adott.",
    },
    "rec_black_video_title":  {"en": "Black recording", "hu": "Fekete felvétel"},
    "rec_black_video_body": {
        "en": "The recording was saved to {path}, but its video track is "
              "completely black. Open Settings → Recording, switch the Windows "
              "capture mode, and use \"Test capture\" before recording again.",
        "hu": "A felvétel elmentve ide: {path}, de a videósáv teljesen fekete. "
              "Nyisd meg a Beállítások → Rögzítés lapot, válts Windows "
              "rögzítési módot, és használd a „Rögzítés tesztelése” gombot, "
              "mielőtt újra felveszel.",
    },

    "gdrive_account_group":   {"en": "Google account", "hu": "Google fiók"},
    "gdrive_account_none":    {"en": "(no account signed in)",
                               "hu": "(nincs bejelentkezett fiók)"},
    "gdrive_account_signed_in": {"en": "Signed in as: {email}",
                                 "hu": "Bejelentkezve: {email}"},
    "gdrive_add_account_btn": {"en": "Add Google account …",
                               "hu": "Google fiók hozzáadása …"},
    "gdrive_signout_btn":     {"en": "Sign out", "hu": "Kijelentkezés"},
    "gdrive_signin_in_progress": {"en": "Opening browser for Google sign-in …",
                                  "hu": "Böngésző megnyitása a Google bejelentkezéshez …"},
    "gdrive_signin_ok":       {"en": "Signed in as {email}.",
                               "hu": "Sikeres bejelentkezés: {email}."},
    "gdrive_signin_failed":   {"en": "Google sign-in failed: {error}",
                               "hu": "Google bejelentkezés sikertelen: {error}"},
    "gdrive_signin_cancelled": {"en": "Sign-in cancelled by user.",
                                "hu": "A bejelentkezést a felhasználó megszakította."},
    "gdrive_signout_confirm_title": {"en": "Sign out", "hu": "Kijelentkezés"},
    "gdrive_signout_confirm_msg":   {
        "en": "Remove the stored credentials for {email}?\n"
              "You will need to sign in again to use Google Drive mode.",
        "hu": "Eltávolítod {email} mentett bejelentkezését?\n"
              "Az újbóli használathoz ismét be kell jelentkezned.",
    },

    "gdrive_oauth_not_configured_title": {
        "en": "Google OAuth not configured",
        "hu": "Google OAuth nincs beállítva",
    },
    "gdrive_oauth_not_configured_msg": {
        "en": "The Google OAuth client_id has not been set in this build. "
              "Replace the placeholder in app/gdrive/oauth_config.py before "
              "using Google Drive mode.",
        "hu": "A Google OAuth client_id nincs beállítva ebben a build-ben. "
              "A Google Drive mód használata előtt cseréld le a "
              "placeholder értékeket az app/gdrive/oauth_config.py-ban.",
    },

    "gdrive_folder_group":    {"en": "Project folder on Drive",
                               "hu": "Projektmappa a Drive-on"},
    "gdrive_folder_none":     {"en": "(no project folder selected)",
                               "hu": "(nincs projektmappa kiválasztva)"},
    "gdrive_pick_folder_btn": {"en": "Choose Drive folder …",
                               "hu": "Drive mappa kiválasztása …"},
    "gdrive_clear_folder_btn": {"en": "Clear", "hu": "Törlés"},
    "gdrive_folder_id_hint":  {"en": "Drive folder ID:",
                               "hu": "Drive mappa azonosító:"},
    "gdrive_pick_folder_title": {"en": "Choose Google Drive Folder",
                                 "hu": "Google Drive mappa kiválasztása"},
    "gdrive_pick_folder_intro": {
        "en": "Paste a Google Drive folder URL or ID. You can copy a URL "
              "from the Drive browser address bar.",
        "hu": "Illessz be egy Google Drive mappa URL-t vagy azonosítót. "
              "Az URL-t a Drive böngésző címsorából másolhatod ki.",
    },
    "gdrive_pick_folder_placeholder": {
        "en": "https://drive.google.com/drive/folders/…  or  folder ID",
        "hu": "https://drive.google.com/drive/folders/…  vagy  mappa ID",
    },
    "gdrive_folder_invalid":  {"en": "That does not look like a Drive folder.",
                               "hu": "Ez nem egy Drive mappának tűnik."},
    "gdrive_folder_unreachable": {
        "en": "Could not reach folder: {error}",
        "hu": "A mappa nem érhető el: {error}",
    },
    "gdrive_folder_selected": {"en": "Folder selected: {name}",
                               "hu": "Mappa kiválasztva: {name}"},

    "gdrive_mode_group":      {"en": "Drive mode — ON / OFF", "hu": "Drive mód — BE / KI"},
    "gdrive_mode_toggle":     {
        "en": "ON — Use Google Drive (only Drive images visible and processed)",
        "hu": "BE — Google Drive használata (csak Drive képek láthatók és feldolgozottak)",
    },
    "gdrive_mode_tip":        {
        "en": "ON: The app connects to Google Drive on startup, downloads the "
              "project database and processes only Drive images. Local folder "
              "is ignored.\n"
              "OFF: Only local images from the selected folder are used. "
              "Drive is not touched.\n"
              "Requires account + project folder to be configured above.",
        "hu": "BE: Az alkalmazás induláskor csatlakozik a Google Drive-hoz, "
              "letölti a projekt adatbázisát, és csak Drive-os képekkel dolgozik. "
              "A helyi mappa figyelmen kívül marad.\n"
              "KI: Csak a kiválasztott helyi mappában lévő képek kerülnek "
              "feldolgozásra. A Drive nincs érintve.\n"
              "Szükséges: fiók + projektmappa megadása fent.",
    },
    "gdrive_db_sync_toggle":  {"en": "Sync database with Drive automatically",
                               "hu": "Adatbázis automatikus szinkronizálása "
                                     "a Drive-val"},

    "gdrive_status_group":    {"en": "Status", "hu": "Állapot"},
    "gdrive_status_offline":  {"en": "Status: offline — sign-in disabled",
                               "hu": "Állapot: offline — bejelentkezés nem elérhető"},
    "gdrive_status_online":   {"en": "Status: online",
                               "hu": "Állapot: online"},
    "gdrive_status_incomplete": {"en": "Pick an account and a project folder to enable Drive mode.",
                                 "hu": "Válassz fiókot és projektmappát a Drive mód aktiválásához."},
    "gdrive_last_sync":       {"en": "Last sync: {when}",
                               "hu": "Utolsó szinkron: {when}"},
    "gdrive_last_sync_never": {"en": "Last sync: never",
                               "hu": "Utolsó szinkron: még nem volt"},
    "gdrive_sync_now_btn":    {"en": "Sync now", "hu": "Szinkronizálás most"},
    "gdrive_cache_clear_btn": {"en": "Clear cache now",
                               "hu": "Cache törlése most"},
    "gdrive_cache_cleared":   {"en": "Cache cleared ({n} file(s) removed).",
                               "hu": "Cache törölve ({n} fájl)."},

    "gdrive_error_title":     {"en": "Google Drive Error",
                               "hu": "Google Drive hiba"},
    "gdrive_info_title":      {"en": "Google Drive",
                               "hu": "Google Drive"},

    # ── Main-window Drive toolbar / status chip ───────────────────────────────
    "gdrive_open_project_btn":   {"en": "Drive Project …",
                                  "hu": "Drive projekt …"},
    "gdrive_open_project_tip":   {
        "en": "Open a Google Drive folder as the active project.\n"
              "The database is downloaded from Drive; changes sync back on save.",
        "hu": "Nyiss meg egy Google Drive mappát aktív projektként.\n"
              "Az adatbázis letöltődik, a változások szinkronizálódnak.",
    },
    "gdrive_close_project_btn":  {"en": "Close Drive",
                                  "hu": "Drive bezárása"},
    "gdrive_close_project_tip":  {
        "en": "Upload pending changes and release the project lock.",
        "hu": "Feltölti a változásokat és feloldja a projektzárat.",
    },
    "gdrive_chip_idle":          {"en": "Drive: —",     "hu": "Drive: —"},
    "gdrive_chip_opening":       {"en": "Drive: connecting …",
                                  "hu": "Drive: csatlakozás …"},
    "gdrive_chip_open":          {"en": "Drive: {name}", "hu": "Drive: {name}"},
    "gdrive_chip_syncing":       {"en": "Drive: syncing …",
                                  "hu": "Drive: szinkronizálás …"},
    "gdrive_chip_error":         {"en": "Drive: error",  "hu": "Drive: hiba"},
    "gdrive_chip_closing":       {"en": "Drive: closing …",
                                  "hu": "Drive: bezárás …"},
    "gdrive_open_failed":        {"en": "Could not open Drive project:\n{error}",
                                  "hu": "Nem sikerült megnyitni a Drive projektet:\n{error}"},
    "gdrive_project_opened":     {"en": "Google Drive project opened: {name}",
                                  "hu": "Google Drive projekt megnyitva: {name}"},
    "gdrive_confirm_close_title": {"en": "Close Drive Project",
                                   "hu": "Drive projekt bezárása"},
    "gdrive_confirm_close_msg":  {
        "en": "Upload pending changes and close the Drive project?\n"
              "The local database cache will be deleted.",
        "hu": "Feltölti a változásokat és bezárja a Drive projektet?\n"
              "A helyi adatbázis gyorsítótár törlődik.",
    },
    "gdrive_closing_wait":       {"en": "Closing Google Drive session …",
                                  "hu": "Google Drive munkamenet bezárása …"},
    "gdrive_not_configured_title": {"en": "Drive Not Configured",
                                    "hu": "Drive nincs konfigurálva"},
    "gdrive_not_configured_msg": {
        "en": "Open Settings → Google Drive and sign in, then pick a project folder.",
        "hu": "Nyisd meg a Beállítások → Google Drive fület, jelentkezz be, "
              "majd válassz project mappát.",
    },
    "ibp_drive_downloading": {
        "en": "⬇ Downloading from Google Drive …",
        "hu": "⬇ Letöltés a Google Drive-ról …",
    },
    "ibp_drive_download_failed": {
        "en": "⚠ Drive download failed: {error}",
        "hu": "⚠ Drive letöltés sikertelen: {error}",
    },
    "gdrive_mode_enabled_opening": {
        "en": "Google Drive mode is ON — connecting to project …",
        "hu": "Google Drive mód BE van kapcsolva — csatlakozás a projekthez …",
    },
    "gdrive_scan_no_session": {
        "en": "Google Drive mode is ON but the project is not open yet.\n"
              "Wait for the Drive connection to finish, then scan again.",
        "hu": "A Google Drive mód be van kapcsolva, de a projekt még nincs megnyitva.\n"
              "Várj, amíg a Drive kapcsolat felépül, majd próbáld újra a beolvasást.",
    },

    # ── Universal search bar ─────────────────────────────────────────────────
    "search_universal_placeholder": {
        "en": "Search persons, dates, places, files…",
        "hu": "Keresés: személyek, dátumok, helyek, fájlok…",
    },
    "search_token_person":      {"en": "Person",    "hu": "Személy"},
    "search_token_nickname":    {"en": "Nickname",  "hu": "Becenév"},
    "search_token_family_code": {"en": "Family ID", "hu": "Csal. azon."},
    "search_token_place":       {"en": "Place",     "hu": "Hely"},
    "search_token_date":        {"en": "Date",      "hu": "Dátum"},
    "search_token_image":       {"en": "Image",     "hu": "Képadat"},
    "search_token_object":      {"en": "Object",    "hu": "Objektum"},
    "search_only_person_cb": {
        "en": "Only this person",
        "hu": "Csak az adott személy",
    },
    "search_names_detail_label": {"en": "Names", "hu": "Nevek"},
    "search_names_detail_placeholder": {
        "en": "Names separated by commas, e.g. Benedek, Matyi",
        "hu": "Nevek vesszővel elválasztva, pl. Benedek, Matyi",
    },
    "ibp_universal_placeholder": {
        "en": "Filter by filename, folder, place, date, person…",
        "hu": "Szűrés fájlnév, mappa, hely, dátum, személy szerint…",
    },
    "ibp_search_results_hdr": {
        "en": "Search results ({n})",
        "hu": "Keresési találatok ({n})",
    },
    "ibp_search_no_results": {
        "en": "No matching images.",
        "hu": "Nincs megfelelő kép.",
    },

}
