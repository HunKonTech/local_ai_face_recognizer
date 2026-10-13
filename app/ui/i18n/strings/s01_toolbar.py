"""UI strings (EN + HU) — part 1, starting at: Toolbar."""

from __future__ import annotations

from typing import Dict

STRINGS: Dict[str, Dict[str, str]] = {
    # ── Toolbar ───────────────────────────────────────────────────────────────
    "select_folder":      {"en": "Select Folder …", "hu": "Mappa kiválasztása …"},
    "manage_folders":     {"en": "Manage Folders …", "hu": "Mappák kezelése …"},
    "no_folder":          {"en": "(no folder selected)", "hu": "(nincs mappa kiválasztva)"},
    "folders_count":      {"en": "{n} folder(s) selected", "hu": "{n} mappa kiválasztva"},
    # ── FolderListDialog ──────────────────────────────────────────────────────
    "folder_list_title":  {"en": "Source Folders", "hu": "Forrás mappák"},
    "folder_list_hint":   {
        "en": "Add one or more folders. All sub-folders are scanned automatically.",
        "hu": "Adj hozzá egy vagy több mappát. Az almappák automatikusan beolvasásra kerülnek.",
    },
    "folder_list_add":    {"en": "Add Folder …", "hu": "Mappa hozzáadása …"},
    "folder_list_remove": {"en": "Remove Selected", "hu": "Kijelölt törlése"},
    "scan_index":    {"en": "Scan & Index", "hu": "Beolvasás és indexelés"},
    "stop":          {"en": "Stop", "hu": "Leállítás"},
    "export_csv":    {"en": "Export CSV", "hu": "CSV exportálás"},
    "export_images": {"en": "Export Images", "hu": "Képek exportálása"},
    "settings":      {"en": "Settings", "hu": "Beállítások"},
    "tpu_status":    {"en": "TPU Status", "hu": "TPU Állapot"},

    # ── Sidebar / action row ──────────────────────────────────────────────────
    "rename_person":      {"en": "Rename Person", "hu": "Személy átnevezése"},
    "merge_into":         {"en": "Merge Into …", "hu": "Összevonás …"},
    "delete_person":      {"en": "Delete Person", "hu": "Személy törlése"},
    "ignore_person":      {"en": "Ignore Forever", "hu": "Kizárás végleg"},
    "ignore_person_tip":  {"en": "Permanently exclude this Unknown person from recognition.\n"
                                 "The face signatures are remembered, so the person will\n"
                                 "not reappear after a re-scan.",
                           "hu": "Az ismeretlen személy végleges kizárása a felismerésből.\n"
                                 "Az arc-jellemzők megmaradnak, így a személy újrafuttatás\n"
                                 "után sem jelenik meg újra."},
    "ignore_person_title": {"en": "Ignore Forever", "hu": "Kizárás végleg"},
    "ignore_person_confirm": {
        "en": "Permanently exclude '{name}' ({n} face(s)) from face recognition?\n\n"
              "The face signatures are stored on a persistent ignore list, so this "
              "person will not reappear after future scans.\n"
              "You can revoke this later in the Ignored Faces manager.",
        "hu": "Véglegesen kizárod '{name}' személyt ({n} arc) az arcfelismerésből?\n\n"
              "Az arc-jellemzők egy tartós kizárólistára kerülnek, így ez a személy "
              "a későbbi futtatások után sem jelenik meg újra.\n"
              "A kizárás később visszavonható a Kizárt arcok kezelőben.",
    },
    "ignore_person_named_msg": {
        "en": "Only auto-named (Unknown) persons can be ignored forever.",
        "hu": "Csak automatikusan elnevezett (Ismeretlen) személy zárható ki véglegesen.",
    },
    "ignore_person_status": {
        "en": "'{name}' ignored forever ({n} face signature(s) stored).",
        "hu": "'{name}' véglegesen kizárva ({n} arc-jellemző elmentve).",
    },
    "ignored_faces_title": {"en": "Ignored Faces", "hu": "Kizárt arcok"},
    "ignored_faces_info": {
        "en": "Faces on this list are permanently excluded from recognition: any "
              "newly detected face matching one of these signatures is suppressed "
              "automatically. Revoking an entry lets the face be recognised again "
              "on the next scan.",
        "hu": "A listán lévő arcok véglegesen ki vannak zárva a felismerésből: az "
              "ezekhez hasonló újonnan talált arcokat a rendszer automatikusan "
              "elrejti. Egy bejegyzés visszavonása után az arc a következő "
              "futtatáskor újra felismerhető.",
    },
    "ignored_faces_unignore": {"en": "Revoke Selected", "hu": "Kijelöltek visszavonása"},
    "ignored_faces_unignore_confirm": {
        "en": "Remove {n} entr(ies) from the permanent ignore list?\n"
              "These faces may reappear after the next scan.",
        "hu": "Eltávolítasz {n} bejegyzést a végleges kizárólistáról?\n"
              "Ezek az arcok a következő futtatás után újra megjelenhetnek.",
    },
    "ignored_faces_count": {"en": "{n} ignored face(s)", "hu": "{n} kizárt arc"},
    "ignored_faces_empty": {"en": "(no ignored faces)", "hu": "(nincs kizárt arc)"},
    "ignored_faces_unknown_source": {"en": "(unknown source)", "hu": "(ismeretlen forrás)"},
    "remove_face":        {"en": "Remove Selected Face", "hu": "Kiválasztott arc eltávolítása"},
    "reassign_face":      {"en": "Reassign Face …", "hu": "Arc áthelyezése …"},
    "people_label":       {"en": "People", "hu": "Személyek"},
    "search_placeholder": {"en": "Search person name …", "hu": "Személy neve …"},
    "n_persons":          {"en": "{n} person(s)", "hu": "{n} személy"},

    # ── Window titles ─────────────────────────────────────────────────────────
    "window_title":       {"en": "Face-Local — Offline Face Grouping",
                           "hu": "Face-Local — Offline arcfelismerő"},
    "activity_log":       {"en": "Activity Log", "hu": "Tevékenységnapló"},
    "toggle_log":         {"en": "Log", "hu": "Napló"},
    "main_toolbar":       {"en": "Main", "hu": "Fő eszköztár"},

    # ── Screen recording ──────────────────────────────────────────────────────
    "rec_start_tip":      {"en": "Start screen recording",
                           "hu": "Képernyőrögzítés indítása"},
    "rec_pause_tip":      {"en": "Pause recording", "hu": "Rögzítés szüneteltetése"},
    "rec_resume_tip":     {"en": "Resume recording", "hu": "Rögzítés folytatása"},
    "rec_stop_tip":       {"en": "Stop & save recording",
                           "hu": "Rögzítés leállítása és mentése"},
    "rec_state_idle":         {"en": "Not recording", "hu": "Nincs felvétel"},
    "rec_state_preflight":    {"en": "Checking capture…",
                               "hu": "Rögzítés ellenőrzése…"},
    "rec_state_recording":    {"en": "Recording ●", "hu": "Rögzítés ●"},
    "rec_state_paused":       {"en": "Paused", "hu": "Szüneteltetve"},
    "rec_state_finalizing":   {"en": "Saving…", "hu": "Mentés…"},
    "rec_state_error":        {"en": "Recording error", "hu": "Rögzítési hiba"},
    "rec_person_prefix":      {"en": "Selected person:", "hu": "Kijelölt személy:"},
    "rec_choose_dir":         {"en": "Choose recording folder",
                               "hu": "Rögzítési mappa kiválasztása"},
    "rec_ffmpeg_missing_title": {"en": "ffmpeg not found",
                                 "hu": "Az ffmpeg nem található"},
    "rec_ffmpeg_missing_body": {
        "en": "Screen recording needs the ffmpeg binary. Install it (e.g. "
              "'brew install ffmpeg' on macOS) or set its path in Settings → "
              "Recording.",
        "hu": "A képernyőrögzítéshez szükséges az ffmpeg program. Telepítsd "
              "(pl. macOS-en 'brew install ffmpeg'), vagy add meg az "
              "elérési útját a Beállítások → Rögzítés alatt.",
    },
    "rec_error_title":        {"en": "Recording error", "hu": "Rögzítési hiba"},
    "rec_saved_title":        {"en": "Recording saved", "hu": "Felvétel elmentve"},
    "rec_saved_body":         {"en": "Saved to:\n{path}",
                               "hu": "Elmentve ide:\n{path}"},
    "rec_quality_low":        {"en": "Low", "hu": "Alacsony"},
    "rec_quality_normal":     {"en": "Normal", "hu": "Normál"},
    "rec_quality_better":     {"en": "Better", "hu": "Jobb"},
    "rec_privacy_title":      {"en": "Recording is about to start",
                               "hu": "A rögzítés mindjárt elindul"},
    "rec_privacy_body": {
        "en": "Screen and audio recording will begin. In an online meeting, "
              "make sure the participants are aware that the session is being "
              "recorded.",
        "hu": "A képernyő és a hang rögzítése elindul. Online megbeszélés "
              "esetén győződj meg róla, hogy a résztvevők tudnak a "
              "rögzítésről.",
    },
    "rec_privacy_dont_ask":   {"en": "Don't show this again",
                               "hu": "Ne jelenjen meg többször"},
    "rec_audio_meter_label":  {"en": "Audio", "hu": "Hang"},
    "rec_no_audio_warning": {
        "en": "The recording contains NO audio track. Check the microphone "
              "permission (System Settings → Privacy → Microphone) and, for "
              "system sound, that a loopback device (e.g. BlackHole) is "
              "installed and selected in Settings → Recording.",
        "hu": "A felvétel NEM tartalmaz hangsávot. Ellenőrizd a mikrofon "
              "jogosultságot (Rendszerbeállítások → Adatvédelem → Mikrofon), "
              "rendszerhanghoz pedig azt, hogy van-e telepített loopback eszköz "
              "(pl. BlackHole), és ki van-e választva a Beállítások → Rögzítés "
              "alatt.",
    },
    "rec_no_audio_warning_windows": {
        "en": "The recording contains NO audio track. Check that desktop apps "
              "are allowed to use the microphone (Settings → Privacy & security "
              "→ Microphone → \"Let desktop apps access your microphone\"), and "
              "that a microphone is selected and not muted in Settings → "
              "Recording. For system sound, enable \"Stereo Mix\" in the Sound "
              "control panel (Recording tab → right-click → Show Disabled "
              "Devices) or install a virtual loopback (e.g. VB-Audio Virtual "
              "Cable), then select it in Settings → Recording.",
        "hu": "A felvétel NEM tartalmaz hangsávot. Ellenőrizd, hogy az asztali "
              "alkalmazások használhatják-e a mikrofont (Beállítások → "
              "Adatvédelem és biztonság → Mikrofon → „Asztali alkalmazások "
              "hozzáférhetnek a mikrofonhoz”), és hogy a Beállítások → Rögzítés "
              "alatt ki van-e választva egy mikrofon, és nincs-e némítva. "
              "Rendszerhanghoz engedélyezd a „Stereo Mix” eszközt a Hang "
              "vezérlőpulton (Felvétel fül → jobb klikk → Letiltott eszközök "
              "megjelenítése), vagy telepíts egy virtuális loopback eszközt "
              "(pl. VB-Audio Virtual Cable), majd válaszd ki a Beállítások → "
              "Rögzítés alatt.",
    },
    "rec_audio_input_device": {"en": "Audio input device (microphone)",
                               "hu": "Hangbemeneti eszköz (mikrofon)"},
    "rec_system_audio_device": {"en": "System audio device (loopback)",
                                "hu": "Rendszerhang eszköz (loopback)"},
    "rec_audio_auto":         {"en": "Automatic", "hu": "Automatikus"},
    "rec_mic_volume":         {"en": "Microphone volume", "hu": "Mikrofon hangereje"},
    "rec_system_volume":      {"en": "System audio volume",
                               "hu": "Rendszerhang hangereje"},
    "rec_mute_microphone":    {"en": "Mute microphone", "hu": "Mikrofon némítása"},
    "rec_mute_system_audio":  {"en": "Mute system audio",
                               "hu": "Rendszerhang némítása"},
    "rec_audio_group":        {"en": "Audio", "hu": "Hang"},
    "tab_face_recognition": {"en": "Face Recognition", "hu": "Arcfelismerés"},
    "tab_image_browser":  {"en": "Image Browser", "hu": "Képböngésző"},
    "tab_family_search":  {"en": "Family Search", "hu": "Családi kereső"},
    "tab_locations":      {"en": "Locations", "hu": "Helyek"},
    "tab_collage":        {"en": "Collage", "hu": "Kollázs"},
    "tab_family_tree":    {"en": "Family Tree", "hu": "Családfa"},

    # ── Family tree panel ──────────────────────────────────────────────────────
    "ft_root":              {"en": "Root:", "hu": "Gyökér:"},
    "ft_choose_root":       {"en": "Focus on person …", "hu": "Személyre fókuszál …"},
    "ft_ctx_focus":         {"en": "Focus on this person", "hu": "Személyre fókuszál"},
    "ft_no_root":        {"en": "(no root selected)", "hu": "(nincs gyökér kiválasztva)"},
    "ft_whole_tree":        {"en": "Whole family tree", "hu": "Teljes családfa"},
    "ft_show_whole":        {"en": "Whole tree", "hu": "Teljes fa"},
    "ft_regenerate":        {"en": "Regenerate from codes", "hu": "Újragenerálás kódból"},
    "ft_regenerate_tip":    {"en": "Re-derive parent/child/spouse links from every person's family code.",
                             "hu": "Szülő/gyerek/házastárs kapcsolatok újraszámolása a családi kódokból."},
    "ft_regenerate_title":  {"en": "Regenerate", "hu": "Újragenerálás"},
    "ft_regenerate_done":   {"en": "Regenerated {n} link(s) from family codes.",
                             "hu": "{n} kapcsolat újragenerálva a családi kódokból."},
    "ft_add_spouse_btn":    {"en": "Add spouse", "hu": "Házastárs hozzáadása"},
    "ft_add_child_btn":     {"en": "Add child", "hu": "Gyerek hozzáadása"},
    "ft_own_code":          {"en": "Own family code:", "hu": "Saját családi kód:"},
    "ft_save_code":         {"en": "Save", "hu": "Mentés"},
    "ft_show_non_members":  {"en": "Show non-members", "hu": "Nem családtagok mutatása"},
    "ft_refresh":           {"en": "Refresh", "hu": "Frissítés"},
    "ft_zoom_in":           {"en": "Zoom in (wheel up / +)", "hu": "Nagyítás (görgő fel / +)"},
    "ft_zoom_out":          {"en": "Zoom out (wheel down / −)", "hu": "Kicsinyítés (görgő le / −)"},
    "ft_zoom_reset":        {"en": "Reset zoom (0)", "hu": "Nagyítás visszaállítása (0)"},
    "ft_select_hint":       {"en": "Select a person in the tree to see their details.",
                             "hu": "Válassz egy személyt a fán a részletekhez."},
    "ft_relation_to_root":  {"en": "Relation to root:", "hu": "Kapcsolat a gyökérhez:"},
    "ft_gender":            {"en": "Gender:", "hu": "Nem:"},
    "ft_nickname":          {"en": "Nickname:", "hu": "Becenév:"},
    "ft_birth":             {"en": "Born:", "hu": "Született:"},
    "ft_birth_place":       {"en": "Birth place:", "hu": "Születési hely:"},
    "ft_death":             {"en": "Died:", "hu": "Elhunyt:"},
    "ft_death_place":       {"en": "Death place:", "hu": "Halálozási hely:"},
    "ft_family_code":       {"en": "Family code:", "hu": "Családi kód:"},
    "ft_edit_person":       {"en": "Edit person …", "hu": "Személy szerkesztése …"},
    "ft_show_images":       {"en": "Show images …", "hu": "Összes kép …"},
    "ft_not_member_badge":  {"en": "(not in family tree)", "hu": "(nem családtag)"},
    "ft_edit_relationships": {"en": "Edit relationships …", "hu": "Kapcsolatok szerkesztése …"},
    "ft_empty_tree":        {"en": "No relationships yet. Use “Edit relationships …” to add some.",
                             "hu": "Még nincsenek kapcsolatok. Add hozzá a „Kapcsolatok szerkesztése …” gombbal."},
    # Relationship kinds (how the selected person relates to the root)
    "ft_kind_self":         {"en": "this is the root", "hu": "ez a gyökér"},
    "ft_kind_spouse":       {"en": "spouse", "hu": "házastárs"},
    "ft_kind_ancestor":     {"en": "ancestor", "hu": "felmenő"},
    "ft_kind_descendant":   {"en": "descendant", "hu": "leszármazott"},
    "ft_kind_sibling":      {"en": "sibling", "hu": "testvér"},
    "ft_kind_cousin":       {"en": "relative", "hu": "rokon"},
    "ft_kind_in_law":       {"en": "in-law", "hu": "házastárs rokona"},
    "ft_kind_unrelated":    {"en": "not related", "hu": "nem rokon"},

    # ── Family tree editor dialog ────────────────────────────────────────────────
    "ft_editor_title":      {"en": "Edit relationships — {name}",
                             "hu": "Kapcsolatok szerkesztése — {name}"},
    "ft_is_member":         {"en": "Show this person in the family tree",
                             "hu": "Szerepeljen a családfában"},
    "ft_current_relationships": {"en": "Current relationships", "hu": "Meglévő kapcsolatok"},
    "ft_add_relationship":  {"en": "Add relationship", "hu": "Kapcsolat hozzáadása"},
    "ft_rel_parent":        {"en": "Parent", "hu": "Szülő"},
    "ft_rel_child":         {"en": "Child", "hu": "Gyermek"},
    "ft_rel_spouse":        {"en": "Spouse", "hu": "Házastárs"},
    "ft_add_btn":           {"en": "Add …", "hu": "Hozzáadás …"},
    "ft_remove_selected":   {"en": "Remove selected", "hu": "Kijelölt törlése"},
    "ft_rel_row":           {"en": "{role}: {name}", "hu": "{role}: {name}"},
    "ft_no_relationships":  {"en": "(no relationships)", "hu": "(nincs kapcsolat)"},
    "ft_pick_person_title": {"en": "Select a person", "hu": "Válassz személyt"},
    "ft_no_persons":        {"en": "No other people to choose from.",
                             "hu": "Nincs más választható személy."},
    "ft_invalid_title":     {"en": "Invalid relationship", "hu": "Érvénytelen kapcsolat"},
    "ft_select_label":      {"en": "Select", "hu": "Kiválasztás"},

    # ── Add-relative dialog ──────────────────────────────────────────────────────
    "ft_add_title":         {"en": "Add {role} — {name}", "hu": "{role} hozzáadása — {name}"},
    "ft_add_existing":      {"en": "Choose an existing person", "hu": "Meglévő személy kiválasztása"},
    "ft_add_new":           {"en": "… or create a new person", "hu": "… vagy új személy létrehozása"},
    "ft_add_new_placeholder": {"en": "New person's name", "hu": "Új személy neve"},
    "ft_add_code_label":    {"en": "Family code (auto-filled):", "hu": "Családi kód (automatikus):"},
    "ft_add_code_hint":     {"en": "Suggested from the family code — you can edit it.",
                             "hu": "A családi kód alapján javasolt — szabadon módosítható."},
    "ft_add_confirm":       {"en": "Add", "hu": "Hozzáadás"},
    "ft_invalid_code_title": {"en": "Invalid family code", "hu": "Érvénytelen családi kód"},
    "ft_link_spouse_too":   {"en": "Also link as a child of {name} (spouse)?",
                             "hu": "Hozzákapcsolódjon {name} (házastárs) gyermekeként is?"},
    "ft_relationships_section": {"en": "Family relationships", "hu": "Családi kapcsolatok"},
    "ft_married_from":      {"en": "Married from:", "hu": "Házasság ettől:"},
    "ft_married_until":     {"en": "until:", "hu": "eddig:"},
    "ft_date_placeholder":  {"en": "e.g. 1954", "hu": "pl. 1954"},
    "ft_rel_row_period":    {"en": "{role}: {name}  ({period})", "hu": "{role}: {name}  ({period})"},
    "ft_period_sep":        {"en": "{start}–{end}", "hu": "{start}–{end}"},
    "ft_marriage_place":      {"en": "Place:", "hu": "Helyszín:"},
    "ft_marriage_place_placeholder": {"en": "e.g. Budapest", "hu": "pl. Budapest"},
    "ft_marriage_edit_title": {"en": "Marriage details — {name}", "hu": "Házasságkötés adatai — {name}"},
    "ft_marriage_edit_hint": {"en": "Tip: double-click a spouse to edit the marriage details.",
                              "hu": "Tipp: kattints duplán egy házastársra a házassági adatok szerkesztéséhez."},
    "ft_spouse_gender_unknown": {
        "en": "Cannot add a spouse: {name}'s biological sex is not set.\n"
              "Please set the biological sex (male/female) first.",
        "hu": "Nem lehet házastársat hozzáadni: {name} biológiai neme nincs megadva.\n"
              "Kérlek, először add meg a biológiai nemet (férfi/nő).",
    },
    "ft_spouse_opposite_sex_required": {
        "en": "Cannot add a spouse: spouses must be of opposite biological sex (one male, one female).\n"
              "The selected person's biological sex is either not set or the same as {name}'s.",
        "hu": "Nem lehet házastársat hozzáadni: a házastársaknak ellentétes biológiai neműeknek kell lenniük (egy férfi, egy nő).\n"
              "A kiválasztott személy biológiai neme nincs megadva, vagy azonos {name} nemével.",
    },
    "ft_spouse_living": {
        "en": "Cannot add a new spouse: {name} is already married to {spouse} who is still alive.\n"
              "Divorce is not supported.",
        "hu": "Nem lehet új házastársat hozzáadni: {name} már házas {spouse} személlyel, aki még él.\n"
              "A válás nem támogatott.",
    },

    "export":             {"en": "Export", "hu": "Export"},
    "collage_import":     {"en": "Collage Import …", "hu": "Kollázs import…"},
    "collage_import_tip": {"en": "Import Picasa collage file (.cxf/.cfx)",
                           "hu": "Picasa kollázs (.cxf/.cfx) beolvasása"},
    "open_collage_file": {"en": "Open Collage File", "hu": "Kollázs fájl megnyitása"},
    "picasa_collage_filter": {"en": "Picasa collage (*.cxf *.cfx);;All files (*)",
                              "hu": "Picasa kollázs (*.cxf *.cfx);;Minden fájl (*)"},
    "extra_search_root": {"en": "Extra search root for resolving Windows paths (optional)",
                          "hu": "Keresési gyökérmappa (opcionális)"},
    "collages_imported": {"en": "{n} collage(s) imported.", "hu": "{n} kollázs importálva."},
    "import_errors":     {"en": "Errors:", "hu": "Hibák:"},
    "collage_import_title": {"en": "Collage Import", "hu": "Kollázs import"},
    "collage_html_export": {"en": "Collage HTML Export", "hu": "Kollázs HTML export"},
    "html_export_folder": {"en": "HTML Export Folder", "hu": "HTML export mappa"},
    "static_site_ready": {"en": "Static website is ready:\n{path}",
                          "hu": "Statikus weboldal elkészült:\n{path}"},
    "check_updates":      {"en": "Check for Updates …", "hu": "Frissítés keresése…"},
    "checking":           {"en": "Checking…", "hu": "Ellenőrzés…"},
    "up_to_date":         {"en": "Up to date", "hu": "Naprakész"},
    "update_available_short": {"en": "Update: v{version}", "hu": "Frissítés: v{version}"},
    "update_status_found": {"en": "New version available: v{version} — click update",
                            "hu": "Új verzió elérhető: v{version} — kattints a frissítésre"},
    "update_notification_title": {"en": "Face-Local update", "hu": "Face-Local frissítés"},
    "update_notification_msg": {"en": "New version available: v{version}. Click to update.",
                                "hu": "Új verzió elérhető: v{version}. Kattints a frissítéshez."},
    "update_connection_failed": {"en": "Could not reach GitHub.\nCheck your internet connection.",
                                 "hu": "Nem sikerült kapcsolódni a GitHub-hoz.\nEllenőrizd az internet kapcsolatot."},
    "app_is_current":     {"en": "The application is up to date.\nCurrent version: v{version}",
                           "hu": "Az alkalmazás naprakész.\nJelenlegi verzió: v{version}"},
    "update_available":   {"en": "New version available", "hu": "Új verzió elérhető"},
    "current":            {"en": "Current", "hu": "Jelenlegi"},
    "latest":             {"en": "Latest", "hu": "Legújabb"},
    "package":            {"en": "Package", "hu": "Csomag"},
    "download_install":   {"en": "Download & Install", "hu": "Letöltés és telepítés"},
    "update_restart":     {"en": "Update & Restart", "hu": "Frissítés és újraindítás"},
    "open_installer":     {"en": "Open installer", "hu": "Telepítő megnyitása"},
    "skip":               {"en": "Skip", "hu": "Kihagyás"},
    "downloading":        {"en": "Downloading…", "hu": "Letöltés…"},
    "downloaded_updating": {"en": "Downloaded — updating & restarting…",
                            "hu": "Letöltve — frissítés és újraindítás…"},
    "downloaded_click_install": {"en": "Downloaded — click to install",
                                 "hu": "Letöltve — kattints a telepítéshez"},
    "force_rescan_tip":   {"en": "Deletes all faces and re-runs detection",
                           "hu": "Törli az összes arcot és újra futtatja a detektálást"},
    "suggestions_tip":    {"en": "Match unknown people against named people",
                           "hu": "Ismeretlen személyek összevetése az elnevezett személyekkel"},
    "person_info":        {"en": "Person Info", "hu": "Személyadatok"},
    "person_info_tip":    {"en": "Edit last name, first name, birth data and notes",
                           "hu": "Vezetéknév, keresztnév, születési adatok és megjegyzés szerkesztése"},
    "gender":             {"en": "Gender", "hu": "Nem"},
    "gender_unknown":     {"en": "Not set", "hu": "Nincs megadva"},
    "gender_male":        {"en": "Male", "hu": "Férfi"},
    "gender_female":      {"en": "Female", "hu": "Nő"},
    "family_code":        {"en": "Family code:", "hu": "Családi azonosító:"},
    "example_family_code": {"en": "e.g. C85", "hu": "pl. C85"},
    "family_code_help": {
        "en": "Examples: C85 = Cikky's 8th child's 5th child, "
              "C80 = spouse of Cikky's 8th child, C0F1 = Cikky's father, "
              "C00T1 = spouse's 1st sibling, C81B = friend of C81.",
        "hu": "Példák: C85 = Cikky 8. gyerekének 5. gyereke, "
              "C80 = Cikky 8. gyerekének házastársa, C0F1 = Cikky apja, "
              "C00T1 = házastárs 1. testvére, C81B = C81 barátja/ismerőse.",
    },
    "family_code_invalid_title": {
        "en": "Invalid family code",
        "hu": "Hibás családi azonosító",
    },
    "external_family_code": {
        "en": "External family code:",
        "hu": "Külső családi azonosító:",
    },
    "example_external_family_code": {"en": "e.g. #Zoli#", "hu": "pl. #Zoli#"},
    "external_family_code_help": {
        "en": "The external person's own family root, e.g. #Zoli# = Zoli, "
              "#Zoli#0 = Zoli's spouse, #Zoli#31 = Zoli's 3rd child's 1st child. "
              "Kept separate from the main family code.",
        "hu": "A külső személy saját családi gyökere, pl. #Zoli# = Zoli, "
              "#Zoli#0 = Zoli házastársa, #Zoli#31 = Zoli 3. gyerekének 1. gyereke. "
              "A fő családi azonosítótól elkülönítve.",
    },
    "external_family_code_invalid_title": {
        "en": "Invalid external family code",
        "hu": "Hibás külső családi azonosító",
    },
    "placeholder_example": {"en": "e.g. {code}", "hu": "pl. {code}"},
    "code_inline_error_tip": {
        "en": "Problem with this code:\n{error}\nIt will be saved exactly as "
              "typed.",
        "hu": "Probléma ezzel a kóddal:\n{error}\nMentéskor pontosan így marad, "
              "ahogy beírtad.",
    },
    "family_code_open_scheme_editor": {
        "en": "Edit family code schemes…",
        "hu": "Családi kód sémák szerkesztése…",
    },
    "family_code_help_scheme": {
        "en": "Active scheme: {name}",
        "hu": "Aktív séma: {name}",
    },
    "family_code_help_editor_hint": {
        "en": "Any code can be saved — one that does not match the scheme only "
              "shows a warning. Use the scheme editor (list icon) to customise "
              "the coding system.",
        "hu": "Bármilyen kód elmenthető — a sémának nem megfelelő kód csak "
              "figyelmeztetést ad. A sémaszerkesztőben (lista ikon) "
              "testreszabhatod a kódrendszert.",
    },

    # ── Family code scheme editor ───────────────────────────────────────────
    "fcs_title": {"en": "Family code schemes", "hu": "Családi kód sémák"},
    "fcs_intro": {
        "en": "A scheme describes how your family codes are built: which letters "
              "start a code (the family roots), which letters mark special "
              "relationships, and which extra notations are allowed. Create your "
              "own scheme, activate it, and share it via export/import — no "
              "programming needed.",
        "hu": "A séma azt írja le, hogyan épülnek fel a családi kódok: melyik "
              "betűkkel kezdődhet a kód (a család gyökerei), melyik betűk jelölik "
              "a különleges kapcsolatokat, és milyen extra jelölések megengedettek. "
              "Készítsd el a saját sémádat, aktiváld, és exporttal/importtal meg is "
              "oszthatod — programozói tudás nem kell hozzá.",
    },
    "fcs_list_label": {"en": "Schemes", "hu": "Sémák"},
    "fcs_active_suffix": {"en": "active", "hu": "aktív"},
    "fcs_builtin_suffix": {"en": "built-in", "hu": "beépített"},
    "fcs_btn_new": {"en": "New", "hu": "Új"},
    "fcs_btn_duplicate": {"en": "Duplicate", "hu": "Másolat"},
    "fcs_btn_delete": {"en": "Delete", "hu": "Törlés"},
    "fcs_btn_activate": {"en": "Set active", "hu": "Aktiválás"},
    "fcs_btn_import": {"en": "Import…", "hu": "Importálás…"},
    "fcs_btn_export": {"en": "Export…", "hu": "Exportálás…"},
    "fcs_btn_save": {"en": "Save scheme", "hu": "Séma mentése"},
    "fcs_saved_status": {"en": "Scheme saved.", "hu": "Séma elmentve."},
    "fcs_builtin_note": {
        "en": "This is the built-in example scheme — it cannot be edited or "
              "deleted. Click 'Duplicate' to create an editable copy for your "
              "own family.",
        "hu": "Ez a beépített példa séma — nem szerkeszthető és nem törölhető. "
              "A 'Másolat' gombbal készíthetsz belőle szerkeszthető változatot a "
              "saját családodra.",
    },
    "fcs_name_label": {"en": "Scheme name:", "hu": "Séma neve:"},
    "fcs_desc_label": {"en": "Description:", "hu": "Leírás:"},
    "fcs_desc_placeholder": {
        "en": "Free-form notes about this coding system (who is who, where it "
              "came from, …)",
        "hu": "Szabad jegyzet erről a kódrendszerről (ki kicsoda, honnan ered, …)",
    },
    "fcs_roots_group": {
        "en": "Root persons — the first letter of every code",
        "hu": "Gyökérszemélyek — minden kód első betűje",
    },
    "fcs_roots_hint": {
        "en": "Every code starts with one of these letters; the letter stands "
              "for the person the family tree is counted from. Example: C = "
              "Cikky, so C85 means Cikky's 8th child's 5th child.",
        "hu": "Minden kód ezen betűk egyikével kezdődik; a betű azt a személyt "
              "jelöli, akitől a családfát számoljuk. Példa: C = Cikky, így a C85 "
              "Cikky 8. gyermekének 5. gyermekét jelenti.",
    },
    "fcs_root_col_letter": {"en": "Letter", "hu": "Betű"},
    "fcs_root_col_name": {"en": "Name", "hu": "Név"},
    "fcs_root_col_note": {"en": "Note", "hu": "Megjegyzés"},
    "fcs_btn_add_root": {"en": "Add root", "hu": "Gyökér hozzáadása"},
    "fcs_btn_remove_root": {"en": "Remove selected", "hu": "Kijelölt törlése"},
    "fcs_markers_group": {
        "en": "Marker letters — special relationships at the end of the code",
        "hu": "Jelölőbetűk — különleges kapcsolatok a kód végén",
    },
    "fcs_markers_hint": {
        "en": "Each marker is a single letter written after the person's number "
              "code. Untick a marker to disable it, or change its letter to "
              "match your own system.",
        "hu": "Mindegyik jelölő egyetlen betű, amely a személy számkódja után áll. "
              "A pipa kivételével kikapcsolhatod, a betű átírásával pedig a saját "
              "rendszeredhez igazíthatod.",
    },
    "fcs_marker_ancestor": {"en": "Ancestors", "hu": "Ősök (felmenők)"},
    "fcs_marker_sibling": {"en": "Siblings", "hu": "Testvérek"},
    "fcs_marker_spouse": {
        "en": "Numbered spouses",
        "hu": "Sorszámozott házastársak",
    },
    "fcs_marker_friend": {
        "en": "Friends / acquaintances",
        "hu": "Barátok / ismerősök",
    },
    "fcs_marker_ancestor_hint": {
        "en": "After the marker: 1 = father, 2 = mother; steps can be chained "
              "(12 = father's mother).",
        "hu": "A jelölő után: 1 = apa, 2 = anya; a lépések láncolhatók "
              "(12 = apai nagymama).",
    },
    "fcs_marker_sibling_hint": {
        "en": "The number selects which sibling; the code may continue with "
              "children or a spouse.",
        "hu": "A szám azt mondja meg, hányadik testvér; a kód folytatódhat "
              "gyermekekkel vagy házastárssal.",
    },
    "fcs_marker_spouse_hint": {
        "en": "Marks a 2nd/3rd/… marriage; digits after it may describe brought "
              "children.",
        "hu": "Második/harmadik/… házasság jelölése; az utána álló számok hozott "
              "gyermekeket írhatnak le.",
    },
    "fcs_marker_friend_hint": {
        "en": "A friend of the given person. Several people may share the same "
              "friend code, so it is never forced to be unique.",
        "hu": "Az adott személy barátja/ismerőse. Ugyanazt a barát-kódot több "
              "személy is kaphatja, ezért sosem kell egyedinek lennie.",
    },
    "fcs_options_group": {"en": "Extra notations", "hu": "További jelölések"},
    "fcs_opt_unlisted_roots": {
        "en": "Accept root letters that are not in the list above",
        "hu": "A fenti listában nem szereplő gyökérbetűk elfogadása",
    },
    "fcs_opt_multi": {
        "en": "Several codes separated by commas — shared friend (C81B,C82B)",
        "hu": "Több kód vesszővel elválasztva — közös barát (C81B,C82B)",
    },
    "fcs_opt_ranges": {
        "en": "Range notation — friend of a whole range (C[1-9]B)",
        "hu": "Tartományjelölés — egy egész tartomány barátja (C[1-9]B)",
    },
    "fcs_opt_braces": {
        "en": "Braces naming the other parent (G1{2}3 = G1's 3rd child from "
              "the 2nd spouse)",
        "hu": "Kapcsos zárójel a másik szülő megadására (G1{2}3 = G1 3. "
              "gyermeke a 2. házastárstól)",
    },
    "fcs_opt_external": {
        "en": "External family codes (#Zoli#31 = the 1st child of Zoli's 3rd "
              "child)",
        "hu": "Külső családi azonosítók (#Zoli#31 = Zoli 3. gyermekének 1. "
              "gyermeke)",
    },
    "fcs_guide_btn": {
        "en": "How does a code work? (step-by-step guide)",
        "hu": "Hogyan épül fel egy kód? (lépésről lépésre)",
    },
    "fcs_guide_digits": {
        "en": "<b>1. First letter</b> — the root person of the family tree "
              "(e.g. {root} = {rootname}).<br>"
              "<b>2. Digits</b> — the descent, step by step:<br>"
              "&nbsp;&nbsp;• {root}0 = the root person, {root}00 = their "
              "spouse<br>"
              "&nbsp;&nbsp;• 1–9 = which child: {root}8 = the 8th child, "
              "{root}85 = the 8th child's 5th child<br>"
              "&nbsp;&nbsp;• 0 at the end = that person's spouse: {root}80<br>"
              "&nbsp;&nbsp;• a digit after a 0 = the spouse's child from an "
              "earlier relationship: {root}805",
        "hu": "<b>1. Kezdőbetű</b> — a családfa gyökérszemélye "
              "(pl. {root} = {rootname}).<br>"
              "<b>2. Számjegyek</b> — a leszármazás lépésről lépésre:<br>"
              "&nbsp;&nbsp;• {root}0 = maga a gyökérszemély, {root}00 = a "
              "házastársa<br>"
              "&nbsp;&nbsp;• 1–9 = hányadik gyermek: {root}8 = a 8. gyermek, "
              "{root}85 = a 8. gyermek 5. gyermeke<br>"
              "&nbsp;&nbsp;• 0 a kód végén = az adott személy házastársa: "
              "{root}80<br>"
              "&nbsp;&nbsp;• 0 után újabb szám = a házastárs korábbi "
              "kapcsolatból hozott gyermeke: {root}805",
    },
    "fcs_guide_markers_title": {
        "en": "<b>3. Marker letter at the end</b> (when enabled):",
        "hu": "<b>3. Jelölőbetű a kód végén</b> (ha engedélyezett):",
    },
    "fcs_guide_marker_anc": {
        "en": "&nbsp;&nbsp;• {letter} = ancestors: 1 = father, 2 = mother, "
              "chainable — {example}",
        "hu": "&nbsp;&nbsp;• {letter} = felmenők: 1 = apa, 2 = anya, "
              "láncolható — {example}",
    },
    "fcs_guide_marker_sib": {
        "en": "&nbsp;&nbsp;• {letter} = sibling — {example}",
        "hu": "&nbsp;&nbsp;• {letter} = testvér — {example}",
    },
    "fcs_guide_marker_spo": {
        "en": "&nbsp;&nbsp;• {letter} = numbered spouse — {example}",
        "hu": "&nbsp;&nbsp;• {letter} = többedik házastárs — {example}",
    },
    "fcs_guide_marker_fri": {
        "en": "&nbsp;&nbsp;• {letter} = friend/acquaintance — {example}",
        "hu": "&nbsp;&nbsp;• {letter} = barát/ismerős — {example}",
    },
    "fcs_guide_extras_title": {
        "en": "<b>4. Extra notations</b> (when enabled):",
        "hu": "<b>4. Extra jelölések</b> (ha engedélyezettek):",
    },
    "fcs_tester_group": {"en": "Try it out", "hu": "Kipróbálás"},
    "fcs_tester_hint": {
        "en": "Type a code and see immediately what it means under this scheme:",
        "hu": "Írj be egy kódot, és azonnal látod, mit jelent ebben a sémában:",
    },
    "fcs_tester_empty": {
        "en": "Type a code above to test it.",
        "hu": "Írj be egy kódot fent a kipróbáláshoz.",
    },
    "fcs_tester_ok": {"en": "✓ Valid: {desc}", "hu": "✓ Érvényes: {desc}"},
    "fcs_tester_bad": {"en": "✗ {error}", "hu": "✗ {error}"},
    "fcs_examples_label": {
        "en": "Examples with the current letters:",
        "hu": "Példák a jelenlegi betűkkel:",
    },
    "fcs_unsaved_title": {"en": "Unsaved changes", "hu": "Nem mentett módosítások"},
    "fcs_unsaved_msg": {
        "en": "Discard the changes made to scheme '{name}'?",
        "hu": "Elveted a(z) '{name}' sémán végzett módosításokat?",
    },
    "fcs_discard_btn": {"en": "Discard changes", "hu": "Módosítások elvetése"},
    "fcs_keep_btn": {"en": "Keep editing", "hu": "Szerkesztés folytatása"},
    "fcs_delete_title": {"en": "Delete scheme", "hu": "Séma törlése"},
    "fcs_delete_msg": {
        "en": "Delete scheme '{name}'? This cannot be undone.",
        "hu": "Törlöd a(z) '{name}' sémát? Ez nem vonható vissza.",
    },
    "fcs_problems_title": {
        "en": "The scheme cannot be saved yet",
        "hu": "A séma még nem menthető",
    },
    "fcs_problems_msg": {
        "en": "Please fix the following:\n\n{problems}",
        "hu": "Kérlek, javítsd a következőket:\n\n{problems}",
    },
    "fcs_import_title": {"en": "Import scheme", "hu": "Séma importálása"},
    "fcs_export_title": {"en": "Export scheme", "hu": "Séma exportálása"},
    "fcs_file_filter": {
        "en": "Family code scheme (*.json)",
        "hu": "Családi kód séma (*.json)",
    },
    "fcs_import_error_title": {"en": "Import failed", "hu": "Sikertelen importálás"},
    "fcs_export_done_msg": {
        "en": "Scheme exported to:\n{path}",
        "hu": "A séma exportálva ide:\n{path}",
    },
    "fcs_new_scheme_name": {"en": "My family", "hu": "Saját családom"},
    "persons_schemes_btn": {
        "en": "Family code schemes…",
        "hu": "Családi kód sémák…",
    },

}
