"""UI strings (EN + HU) — part 2, starting at: Family search."""

from __future__ import annotations

from typing import Dict

STRINGS: Dict[str, Dict[str, str]] = {
    # ── Family search ───────────────────────────────────────────────────────
    "family_person_a":    {"en": "First person", "hu": "Első személy"},
    "family_person_b":    {"en": "Second person", "hu": "Második személy"},
    "family_relationship_filter": {"en": "Relationship", "hu": "Kapcsolat"},
    "family_search_btn":  {"en": "Search", "hu": "Keresés"},
    "family_names_placeholder": {
        "en": "Names separated by commas, e.g. Benedek, Matyi, Domi",
        "hu": "Nevek vesszővel elválasztva, pl. Benedek, Matyi, Domi",
    },
    "family_allow_others_yes": {"en": "Others allowed", "hu": "Lehetnek mások is"},
    "family_allow_others_no": {"en": "Only these people", "hu": "Csak ezek a személyek"},
    "family_details_closed": {"en": "Detailed search", "hu": "Részletes kereső"},
    "family_details_open": {"en": "Detailed search", "hu": "Részletes kereső"},
    "family_detail_person_text": {"en": "Person data", "hu": "Személyadat"},
    "family_detail_person_placeholder": {
        "en": "Name, nickname, birth/death data, notes",
        "hu": "Név, becenév, születési/halálozási adat, megjegyzés",
    },
    "family_detail_image_text": {"en": "Image data", "hu": "Képadat"},
    "family_detail_image_placeholder": {
        "en": "Filename, path, date or place",
        "hu": "Fájlnév, elérési út, dátum vagy hely",
    },
    "family_detail_photo_date": {"en": "Photo date", "hu": "Kép dátuma"},
    "family_detail_place_text": {"en": "Place", "hu": "Hely"},
    "family_detail_place_placeholder": {"en": "Place name", "hu": "Hely neve"},
    "family_gender_any": {"en": "Any gender", "hu": "Bármelyik nem"},
    "family_mode_both":   {"en": "May include others", "hu": "Lehetnek mások is"},
    "family_mode_exact":  {"en": "Only these two", "hu": "Csak ez a két személy"},
    "family_filter_any":  {"en": "Any relationship", "hu": "Bármilyen kapcsolat"},
    "family_filter_spouse": {"en": "Spouse", "hu": "Házastárs"},
    "family_filter_parent": {"en": "First is parent", "hu": "Első a szülő"},
    "family_filter_child": {"en": "First is child", "hu": "Első a gyerek"},
    "family_filter_sibling": {"en": "Sibling", "hu": "Testvér"},
    "family_add_relationship": {"en": "Save relationship:", "hu": "Kapcsolat mentése:"},
    "family_save_relationship": {"en": "Save", "hu": "Mentés"},
    "family_add_spouse": {"en": "Spouses", "hu": "Házastársak"},
    "family_add_a_parent_b": {"en": "First is parent of second", "hu": "Első szülője a másodiknak"},
    "family_add_b_parent_a": {"en": "Second is parent of first", "hu": "Második szülője az elsőnek"},
    "family_select_two": {"en": "Select two different people.", "hu": "Válassz ki két különböző személyt."},
    "family_no_relationship": {"en": "No stored or derived relationship between them.",
                               "hu": "Nincs tárolt vagy számított kapcsolat közöttük."},
    "family_validation_title": {"en": "Invalid relationship", "hu": "Hibás kapcsolat"},
    "family_prev":       {"en": "Previous", "hu": "Előző"},
    "family_next":       {"en": "Next", "hu": "Következő"},
    "family_results_empty": {"en": "No results", "hu": "Nincs találat"},
    "family_results_range": {"en": "{start}-{end} of {total} images",
                             "hu": "{start}-{end} / {total} kép"},
    "family_no_results": {"en": "No matching images.", "hu": "Nincs megfelelő kép."},
    "family_result_item": {"en": "{name}  ({persons} people, {faces} faces)",
                           "hu": "{name}  ({persons} személy, {faces} arc)"},
    "ibp_person_filter_hdr": {"en": "Person filter", "hu": "Személy szűrő"},
    "ibp_person_filter_apply": {"en": "Apply", "hu": "Alkalmaz"},
    "ibp_person_filter_clear": {"en": "Clear", "hu": "Törlés"},

    # ── Dialogs — general ─────────────────────────────────────────────────────
    "ok":         {"en": "OK", "hu": "OK"},
    "cancel":     {"en": "Cancel", "hu": "Mégse"},
    "yes":        {"en": "Yes", "hu": "Igen"},
    "no":         {"en": "No", "hu": "Nem"},
    "close":      {"en": "Close", "hu": "Bezárás"},
    "clear":      {"en": "Clear", "hu": "Törlés"},
    "ready":      {"en": "Ready", "hu": "Kész"},
    "dep_installing":   {"en": "Installing AI backend: {name}…", "hu": "AI backend telepítése: {name}…"},
    "dep_installed":    {"en": "AI backend installed: {name} ✓ (active on next scan)", "hu": "AI backend telepítve: {name} ✓ (következő szkenneléskor aktív)"},
    "dep_install_done": {"en": "AI backend ready ✓", "hu": "AI backend kész ✓"},

    # ── AI Packages settings tab ──────────────────────────────────────────────
    "settings_tab_packages":  {"en": "AI Packages", "hu": "AI csomagok"},
    "pkg_tab_intro":          {"en": "Optional AI packages improve face recognition quality. "
                                     "Missing packages are installed via pip into your Python "
                                     "environment. No admin rights required.",
                               "hu": "Az opcionális AI csomagok javítják az arcfelismerés "
                                     "minőségét. A hiányzó csomagok pip-pel települnek a Python "
                                     "környezetedbe. Rendszergazdai jog nem szükséges."},
    "pkg_group_embedding":    {"en": "Embedding backend (recognition quality)", "hu": "Beágyazás backend (felismerés minősége)"},
    "pkg_group_detection":    {"en": "Detection / Verification", "hu": "Detektálás / Verifikáció"},
    "pkg_detection_note":     {"en": "onnxruntime + insightface enable the SCRFD co-detector that "
                                     "recovers profile and turned faces. The InsightFace buffalo_l "
                                     "model pack (~200 MB) downloads automatically on first scan.",
                               "hu": "Az onnxruntime + insightface engedélyezi az SCRFD "
                                     "ko-detektort, amely profil- és elfordult arcokat talál meg. "
                                     "Az InsightFace buffalo_l modellcsomag (~200 MB) az első "
                                     "szkenneléskor töltődik le automatikusan."},
    "pkg_active_backend":     {"en": "Active embedding backend: {backend}", "hu": "Aktív beágyazás backend: {backend}"},
    "pkg_emb_model_label":    {"en": "Embedding model:", "hu": "Beágyazás modell:"},
    "pkg_emb_mobilefacenet":  {"en": "MobileFaceNet (default, lightweight, 192-dim)",
                               "hu": "MobileFaceNet (alapértelmezett, könnyű, 192-dim)"},
    "pkg_emb_arcface":        {"en": "ArcFace (ResNet-50) — higher quality, higher resource use (512-dim)",
                               "hu": "ArcFace (ResNet-50) — magasabb minőség, magasabb erőforrásigény (512-dim)"},
    "pkg_emb_arcface_needs":  {"en": "ArcFace reuses the InsightFace buffalo_l model — requires the insightface + onnxruntime packages below.",
                               "hu": "Az ArcFace az InsightFace buffalo_l modellt használja újra — szükséges hozzá az alábbi insightface + onnxruntime csomag."},
    "pkg_emb_apple_ane":      {"en": "Use the Apple Neural Engine (Core ML) for embeddings",
                               "hu": "Apple Neural Engine (Core ML) használata a beágyazáshoz"},
    "pkg_emb_apple_ane_note": {"en": "Apple Silicon only. Runs MobileFaceNet on the Neural Engine / GPU via Core ML when a converted model (models/mobilefacenet.mlpackage) is present; otherwise the CPU path is used. The ArcFace backend already uses the Neural Engine via Core ML. Current accelerator: {accel}.",
                               "hu": "Csak Apple Silicon. A MobileFaceNet-et a Neural Engine-en / GPU-n futtatja Core ML-lel, ha van konvertált modell (models/mobilefacenet.mlpackage); különben a CPU út fut. Az ArcFace backend már most a Neural Engine-t használja Core ML-en át. Jelenlegi gyorsító: {accel}."},
    "pkg_emb_switch_title":   {"en": "Embedding model changed — re-scan required",
                               "hu": "Beágyazás modell megváltozott — újraszkennelés szükséges"},
    "pkg_emb_switch_msg":     {"en": "The embedding model was changed ({old} → {new}).\n\n"
                                     "The two models produce incompatible vectors, so all existing face "
                                     "embeddings are now stale — recognition will not work correctly "
                                     "until they are regenerated.\n\n"
                                     "What to do:\n"
                                     "  1. Open the AI tab (main window, top menu)\n"
                                     "  2. Click \"AI Rebuild from Scratch\"\n\n"
                                     "This re-detects every face, regenerates all embeddings with the "
                                     "new model, and retrains the recognition network. "
                                     "Your confirmed person assignments are kept.",
                               "hu": "A beágyazás modell megváltozott ({old} → {new}).\n\n"
                                     "A két modell inkompatibilis vektorokat állít elő, ezért a meglévő "
                                     "arc-embeddingek érvénytelenek — a felismerés nem fog helyesen "
                                     "működni, amíg ezek újra nem generálódnak.\n\n"
                                     "Teendő:\n"
                                     "  1. Nyisd meg az AI fület (főablak, felső menü)\n"
                                     '  2. Kattints az „AI újraépítés nulláról” gombra\n\n'
                                     "Ez újradetektál minden arcot, az új modellel újragenerálja az "
                                     "összes embeddinget, és újratanítja a felismerő hálót. "
                                     "A megerősített személy-hozzárendelések megmaradnak."},
    "pkg_backend_none":       {"en": "none (HOG stub — very low quality!)", "hu": "nincs (HOG stub — nagyon gyenge minőség!)"},
    "pkg_model_file_ok":      {"en": "Model file: {path}  ✓", "hu": "Modell fájl: {path}  ✓"},
    "pkg_model_file_missing": {"en": "Model file missing: {path}", "hu": "Modell fájl hiányzik: {path}"},
    "pkg_status_installed":   {"en": "Installed", "hu": "Telepítve"},
    "pkg_status_missing":     {"en": "Not installed", "hu": "Nincs telepítve"},
    "pkg_status_installing":  {"en": "Installing…", "hu": "Telepítés…"},
    "pkg_status_failed":      {"en": "Install failed — retry?", "hu": "Telepítés sikertelen — újra?"},
    "pkg_install_btn":        {"en": "Install", "hu": "Telepítés"},
    "pkg_installing_btn":     {"en": "Installing…", "hu": "Telepítés…"},
    "pkg_install_all_btn":    {"en": "Install all missing", "hu": "Összes hiányzó telepítése"},
    "pkg_refresh_btn":        {"en": "Refresh status", "hu": "Állapot frissítése"},
    "pkg_model_download_btn":    {"en": "Download model", "hu": "Modell letöltése"},
    "pkg_model_retry_btn":       {"en": "Retry download", "hu": "Újra letöltés"},
    "pkg_model_downloading":     {"en": "Downloading mobilefacenet.tflite…", "hu": "mobilefacenet.tflite letöltése…"},
    "pkg_model_download_failed": {"en": "Download failed — check your internet connection.", "hu": "Letöltés sikertelen — ellenőrizd az internetkapcsolatot."},
    "error":      {"en": "Error", "hu": "Hiba"},
    "warning":    {"en": "Warning", "hu": "Figyelmeztetés"},
    "save_error": {"en": "Save Error", "hu": "Mentési hiba"},
    "unknown":    {"en": "Unknown", "hu": "Ismeretlen"},
    "filename":   {"en": "Filename", "hu": "Fájlnév"},
    "path":       {"en": "Path", "hu": "Elérési út"},
    "raw_path":   {"en": "Path (raw)", "hu": "Elérési út (raw)"},
    "persons":    {"en": "Persons", "hu": "Személyek"},
    "no_recognized_face": {"en": "(no recognized face)", "hu": "(nincs felismert arc)"},
    "source_file_missing": {"en": "⚠ Source file not found!", "hu": "⚠ Forrásfájl nem található!"},

    # ── Dialogs — delete person ───────────────────────────────────────────────
    "delete_person_title":   {"en": "Delete Person",
                              "hu": "Személy törlése"},
    "delete_person_confirm": {"en": "Delete '{name}' and all {n} of their faces?\n"
                                    "Every face, bounding box, thumbnail and embedding is "
                                    "permanently removed.\nThis cannot be undone.",
                              "hu": "Biztosan törlöd ezt a személyt ('{name}') és az összes "
                                    "hozzá tartozó arcot ({n} db)?\nMinden arckeret, előnézet "
                                    "és felismerési adat véglegesen törlődik.\n"
                                    "Ez a művelet nem vonható vissza."},
    "delete_person_ignore_check": {
        "en": "Also exclude these faces from recognition forever",
        "hu": "Az arcok végleges kizárása a felismerésből is"},
    "delete_person_status": {"en": "Deleted '{name}': {n} face(s) removed.",
                             "hu": "'{name}' törölve: {n} arc eltávolítva."},
    "delete_person_status_ignored": {
        "en": "Deleted '{name}': {n} face(s) removed, {k} excluded forever.",
        "hu": "'{name}' törölve: {n} arc eltávolítva, {k} véglegesen kizárva."},

    # ── Dialogs — scan ────────────────────────────────────────────────────────
    "no_folder_title":   {"en": "No Folder", "hu": "Nincs mappa"},
    "no_folder_msg":     {"en": "Please select a root folder first.",
                          "hu": "Először válasszon ki egy mappát."},
    "busy_title":        {"en": "Busy", "hu": "Foglalt"},
    "busy_msg":          {"en": "A scan is already running.", "hu": "Már fut egy beolvasás."},

    # ── Dialogs — rename ─────────────────────────────────────────────────────
    "empty_name_title":     {"en": "Empty Name", "hu": "Üres név"},
    "empty_name_msg":       {"en": "Name cannot be empty.", "hu": "A név nem lehet üres."},
    "protected_rename_title": {"en": "Cannot Rename", "hu": "Nem átnevezhető"},
    "protected_rename_msg":   {"en": "'Ismeretlen' is a protected category and cannot be renamed.",
                                "hu": "Az 'Ismeretlen' egy védett kategória, nem nevezhető át."},
    "protected_delete_title": {"en": "Cannot Delete", "hu": "Nem törölhető"},
    "protected_delete_msg":   {"en": "'Ismeretlen' is a protected category and cannot be deleted.",
                                "hu": "Az 'Ismeretlen' egy védett kategória, nem törölhető."},

    # ── Dialogs — remove face ─────────────────────────────────────────────────
    "remove_face_title": {"en": "Remove Face", "hu": "Arc eltávolítása"},
    "remove_face_msg":   {"en": "Remove this face from the cluster and exclude it from clustering?",
                          "hu": "Eltávolítja ezt az arcot a csoportból és kizárja az újra csoportosításból?"},

    # ── Dialogs — merge ───────────────────────────────────────────────────────
    "merge_error_title": {"en": "Merge Error", "hu": "Összevonási hiba"},
    "reassign_title":    {"en": "Reassign Face To …", "hu": "Arc áthelyezése …"},
    "merge_source_into": {"en": "Merge <b>{name}</b> into:", "hu": "<b>{name}</b> összevonása ide:"},
    "merge_faces_count": {"en": "{name} ({n} faces)", "hu": "{name} ({n} arc)"},
    "merge_source_deleted": {"en": "The source person will be deleted after merging.",
                             "hu": "A forrás személy az összevonás után törlődik."},
    "rename_person_to":  {"en": "Rename <b>{name}</b> to:", "hu": "<b>{name}</b> új neve:"},

    # ── Export ────────────────────────────────────────────────────────────────
    "export_done":       {"en": "Export Done", "hu": "Exportálás kész"},
    "export_csv_saved":  {"en": "Saved to:\n{path}", "hu": "Mentve:\n{path}"},
    "no_person_title":   {"en": "No Person Selected", "hu": "Nincs személy kiválasztva"},
    "no_person_msg":     {"en": "Select a person in the sidebar first.",
                          "hu": "Először válasszon személyt az oldalsávban."},
    "exported_n":        {"en": "Exported {n} image(s) to:\n{folder}",
                          "hu": "{n} kép exportálva:\n{folder}"},
    "export_title":      {"en": "Export", "hu": "Exportálás"},
    # ── Full project package (.facepack) ──────────────────────────────────────
    "pkg_group":         {"en": "Full project package (.facepack)",
                          "hu": "Teljes projektcsomag (.facepack)"},
    "pkg_export_desc":   {"en": "Bundle the database, face crops, local images and "
                                "config into a single portable .facepack file.",
                          "hu": "Az adatbázis, az arckivágások, a lokális képek és a "
                                "konfiguráció egyetlen hordozható .facepack fájlba."},
    "pkg_export_btn":    {"en": "Export full project", "hu": "Teljes projekt exportálása"},
    "pkg_import_desc":   {"en": "Restore a project from a .facepack file into a new "
                                "project folder.",
                          "hu": "Projekt visszaállítása egy .facepack fájlból új "
                                "projektmappába."},
    "pkg_import_btn":    {"en": "Import project", "hu": "Projekt importálása"},
    "pkg_export_dialog": {"en": "Export project package", "hu": "Projektcsomag exportálása"},
    "pkg_import_dialog": {"en": "Select project package", "hu": "Projektcsomag kiválasztása"},
    "pkg_import_dest":   {"en": "Choose destination project folder",
                          "hu": "Válassz cél projektmappát"},
    "pkg_filter":        {"en": "Face-Local package (*.facepack)",
                          "hu": "Face-Local csomag (*.facepack)"},
    "pkg_progress_export": {"en": "Exporting project package…",
                            "hu": "Projektcsomag exportálása…"},
    "pkg_progress_import": {"en": "Importing project package…",
                            "hu": "Projektcsomag importálása…"},
    "pkg_export_done":   {"en": "Project package", "hu": "Projektcsomag"},
    "pkg_export_ok":     {"en": "Saved {images} image(s) and {crops} crop(s) to:\n{path}",
                          "hu": "{images} kép és {crops} kivágás mentve ide:\n{path}"},
    "pkg_export_warn":   {"en": "\n\nThe export finished, but {missing} image(s) "
                                "were not available and could not be bundled "
                                "(see log).",
                          "hu": "\n\nAz export elkészült, de {missing} kép nem volt "
                                "elérhető, és nem került a csomagba (részletek a logban)."},
    "pkg_import_title":  {"en": "Import project", "hu": "Projekt importálása"},
    "pkg_import_confirm": {"en": "This will extract the project into:\n{dest}\n\n"
                                 "Your current project will not be overwritten. "
                                 "Continue?",
                           "hu": "A projekt ide lesz kibontva:\n{dest}\n\n"
                                 "A jelenlegi projekt nem íródik felül. Folytatja?"},
    "pkg_import_ok":     {"en": "Project imported to:\n{dest}\n\nOpen it now?",
                          "hu": "Projekt importálva ide:\n{dest}\n\nMegnyitja most?"},
    "pkg_import_opened": {"en": "Imported project is now active.",
                          "hu": "Az importált projekt mostantól aktív."},
    "pkg_import_missing": {"en": "\n\nNote: {missing} original image(s) could not "
                                 "be located after import (see log).",
                           "hu": "\n\nMegjegyzés: {missing} eredeti kép nem található "
                                 "az import után (részletek a logban)."},
    "pkg_import_later":  {"en": "Imported project is ready at:\n{dest}",
                          "hu": "Az importált projekt készen áll itt:\n{dest}"},
    "pkg_error_title":   {"en": "Project package error", "hu": "Projektcsomag hiba"},

    # ── Database-only package (.facedb) ───────────────────────────────────────
    "dbpkg_group":       {"en": "Database only (.facedb)",
                          "hu": "Csak adatbázis (.facedb)"},
    "dbpkg_export_desc": {"en": "Export the database alone — no images, no face "
                                "crops. Images are recorded relative to the "
                                "library root, so the other machine can keep its "
                                "own base folder (e.g. C:\\photos here, D:\\photos "
                                "there) as long as the folders below it match.",
                          "hu": "Csak az adatbázis exportálása — képek és "
                                "arckivágások nélkül. A képek a képtár "
                                "gyökeréhez képest tárolódnak, így a másik gépen "
                                "lehet más az alapmappa (pl. itt C:\\kepek, ott "
                                "D:\\kepek), ha az alatta lévő mappaszerkezet "
                                "megegyezik."},
    "dbpkg_export_btn":  {"en": "Export database", "hu": "Adatbázis exportálása"},
    "dbpkg_import_desc": {"en": "Import a .facedb file: pick the base image folder "
                                "on this machine and every path is re-based onto "
                                "it. Images that are still not found can be "
                                "matched by scanning a folder.",
                          "hu": ".facedb fájl importálása: válaszd ki a gépen "
                                "lévő alap képmappát, és minden útvonal ehhez "
                                "igazodik. Az így sem megtalált képek mappa "
                                "beszkennelésével párosíthatók."},
    "dbpkg_import_btn":  {"en": "Import database", "hu": "Adatbázis importálása"},
    "dbpkg_export_dialog": {"en": "Export database package",
                            "hu": "Adatbázis-csomag exportálása"},
    "dbpkg_import_dialog": {"en": "Select database package",
                            "hu": "Adatbázis-csomag kiválasztása"},
    "dbpkg_filter":      {"en": "Face-Local database (*.facedb)",
                          "hu": "Face-Local adatbázis (*.facedb)"},
    "dbpkg_export_done": {"en": "Database package", "hu": "Adatbázis-csomag"},
    "dbpkg_export_ok":   {"en": "{images} image record(s) saved to:\n{path}",
                          "hu": "{images} képrekord mentve ide:\n{path}"},
    "dbpkg_export_warn": {"en": "\n\n{outside} image(s) are stored outside the "
                                "library root, so only their original absolute "
                                "path travels with the package.",
                          "hu": "\n\n{outside} kép a képtár gyökerén kívül van, "
                                "ezért náluk csak az eredeti teljes elérési út "
                                "kerül a csomagba."},
    "dbpkg_export_no_root": {"en": "No image library root is configured, so paths "
                                   "cannot be made portable. Set the library root "
                                   "in Settings first, or continue with absolute "
                                   "paths?",
                             "hu": "Nincs beállítva képtár-gyökér, így az útvonalak "
                                   "nem tehetők hordozhatóvá. Állítsd be a "
                                   "Beállításokban, vagy folytatod teljes elérési "
                                   "utakkal?"},
    "dbpkg_import_root": {"en": "Select the base image folder on THIS machine",
                          "hu": "Válaszd ki a képek alapmappáját EZEN a gépen"},
    "dbpkg_import_dest": {"en": "Choose a folder for the imported database",
                          "hu": "Válassz mappát az importált adatbázisnak"},
    "dbpkg_import_title": {"en": "Import database", "hu": "Adatbázis importálása"},
    "dbpkg_import_confirm": {"en": "The database will be imported into:\n{dest}\n\n"
                                   "Image paths will be re-based onto:\n{root}\n\n"
                                   "Your current project will not be overwritten. "
                                   "Continue?",
                             "hu": "Az adatbázis ide kerül:\n{dest}\n\n"
                                   "A képek útvonala ehhez igazodik:\n{root}\n\n"
                                   "A jelenlegi projekt nem íródik felül. Folytatod?"},
    "dbpkg_import_ok":   {"en": "Database imported to:\n{dest}\n\n"
                                "{resolved} image(s) found, {unresolved} not found.\n\n"
                                "Open it now?",
                          "hu": "Adatbázis importálva ide:\n{dest}\n\n"
                                "{resolved} kép megtalálva, {unresolved} nem "
                                "található.\n\nMegnyitod most?"},
    "dbpkg_import_later": {"en": "The imported database is ready at:\n{dest}",
                           "hu": "Az importált adatbázis készen áll itt:\n{dest}"},
    "dbpkg_import_opened": {"en": "The imported database is now active.",
                            "hu": "Az importált adatbázis mostantól aktív."},
    "dbpkg_import_unresolved": {"en": "\n\n{n} image(s) were not found at the "
                                      "re-based location. Scan a folder now to "
                                      "match them by file name?",
                                "hu": "\n\n{n} kép nem található az igazított "
                                      "helyen. Beszkennelsz most egy mappát, hogy "
                                      "fájlnév alapján párosítsuk őket?"},
    "dbpkg_error_title": {"en": "Database package error", "hu": "Adatbázis-csomag hiba"},

    # ── Missing-image path repair (scan & match) ──────────────────────────────
    "pathfix_group":     {"en": "Missing images", "hu": "Hiányzó képek"},
    "pathfix_desc":      {"en": "Some records point at files that are not on this "
                                "machine? Scan a folder and match them by file "
                                "name and content.",
                          "hu": "Vannak rekordok, amelyek nem létező fájlra "
                                "mutatnak? Szkennelj be egy mappát, és párosítsd "
                                "őket fájlnév és tartalom alapján."},
    "pathfix_open_btn":  {"en": "Find missing images …",
                          "hu": "Hiányzó képek keresése …"},
    "pathfix_title":     {"en": "Find missing images", "hu": "Hiányzó képek keresése"},
    "pathfix_intro":     {"en": "Records whose file is not where the database "
                                "expects it are listed below. The chosen folders "
                                "are scanned and every missing record is matched "
                                "by file name; the recorded content hash decides "
                                "whether a match is certain. Review the rows, then "
                                "apply.",
                          "hu": "Alább azok a rekordok szerepelnek, amelyek fájlja "
                                "nincs ott, ahol az adatbázis szerint lennie "
                                "kellene. A kiválasztott mappák beolvasásra "
                                "kerülnek, és minden hiányzó rekord fájlnév "
                                "alapján párosul; a tárolt tartalom-ujjlenyomat "
                                "dönti el, hogy a találat biztos-e. Nézd át a "
                                "sorokat, majd alkalmazd."},
    "pathfix_folders_label": {"en": "Folders to scan", "hu": "Beolvasandó mappák"},
    "pathfix_add_folder": {"en": "Add folder …", "hu": "Mappa hozzáadása …"},
    "pathfix_remove_folder": {"en": "Remove", "hu": "Eltávolítás"},
    "pathfix_verify_hash": {"en": "Verify by file content (slower, but certain)",
                            "hu": "Ellenőrzés fájltartalommal (lassabb, de biztos)"},
    "pathfix_verify_hash_tip": {
        "en": "Compares the SHA-256 hash recorded at index time with the "
              "candidate file. A match is proof, not a guess.",
        "hu": "Az indexeléskor rögzített SHA-256 ujjlenyomatot hasonlítja össze a "
              "jelölt fájllal. Az egyezés bizonyíték, nem tipp.",
    },
    "pathfix_start_btn": {"en": "Start scan", "hu": "Beolvasás indítása"},
    "pathfix_running":   {"en": "Scanning in the background — see the Task Manager.",
                          "hu": "Beolvasás a háttérben — lásd a Feladatkezelőt."},
    "pathfix_cancelled": {"en": "Scan stopped.", "hu": "Beolvasás leállítva."},
    "pathfix_need_folder": {"en": "Add at least one folder to scan.",
                            "hu": "Adj hozzá legalább egy beolvasandó mappát."},
    "pathfix_no_missing": {"en": "Every image record points at an existing file.",
                           "hu": "Minden képrekord létező fájlra mutat."},
    "pathfix_summary":   {"en": "{missing} missing · {matched} matched "
                                "({confident} certain) · {scanned} file(s) scanned",
                          "hu": "{missing} hiányzó · {matched} párosítva "
                                "({confident} biztos) · {scanned} fájl beolvasva"},
    "pathfix_col_file":  {"en": "File", "hu": "Fájl"},
    "pathfix_col_expected": {"en": "Expected location", "hu": "Várt helye"},
    "pathfix_col_match": {"en": "Match", "hu": "Találat"},
    "pathfix_col_confidence": {"en": "Confidence", "hu": "Biztonság"},
    "pathfix_skip_option": {"en": "— skip —", "hu": "— kihagyás —"},
    "pathfix_browse_option": {"en": "Browse …", "hu": "Tallózás …"},
    "pathfix_conf_proof": {"en": "Certain (content match)",
                           "hu": "Biztos (tartalom egyezik)"},
    "pathfix_conf_mismatch": {"en": "Different content!",
                              "hu": "Eltérő tartalom!"},
    "pathfix_conf_guess": {"en": "{score}% (name)", "hu": "{score}% (név)"},
    "pathfix_conf_none": {"en": "No match", "hu": "Nincs találat"},
    "pathfix_select_confident": {"en": "Select certain",
                                 "hu": "Biztosak kijelölése"},
    "pathfix_select_all": {"en": "Select all", "hu": "Összes kijelölése"},
    "pathfix_select_none": {"en": "Select none", "hu": "Kijelölés törlése"},
    "pathfix_apply_btn": {"en": "Apply selected", "hu": "Kijelöltek alkalmazása"},
    "pathfix_nothing_selected": {"en": "No row is selected for applying.",
                                 "hu": "Egy sor sincs kijelölve alkalmazásra."},
    "pathfix_applied":   {"en": "{n} image(s) re-attached, {skipped} skipped.",
                          "hu": "{n} kép újra hozzárendelve, {skipped} kihagyva."},
    "pathfix_preview_none": {"en": "No preview", "hu": "Nincs előnézet"},

    "deep_model_group":         {"en": "AI recognition model",
                                 "hu": "AI felismerési modell"},
    "deep_model_export_desc":   {"en": "Save the trained model to a file so it can be "
                                 "transferred to another computer or kept as a backup.",
                                 "hu": "A betanított modell mentése fájlba, hogy más "
                                 "számítógépre átvihető vagy biztonsági másolatként "
                                 "megőrizhető legyen."},
    "deep_model_export_btn":    {"en": "Export model", "hu": "Modell exportálása"},
    "deep_model_export_dialog": {"en": "Save AI model", "hu": "AI modell mentése"},
    "deep_model_export_ok":     {"en": "Model saved to:\n{path}", "hu": "Modell mentve:\n{path}"},
    "deep_model_export_none":   {"en": "No trained model found. Run the AI scan first.",
                                 "hu": "Nincs betanított modell. Futtasd először az AI beolvasást."},
    "deep_model_import_desc":   {"en": "Load a previously exported model file. "
                                 "The current model will be replaced immediately.",
                                 "hu": "Korábban exportált modellfájl betöltése. "
                                 "Az aktuális modell azonnal felülíródik."},
    "deep_model_import_btn":    {"en": "Import model", "hu": "Modell importálása"},
    "deep_model_import_dialog": {"en": "Open AI model", "hu": "AI modell megnyitása"},
    "deep_model_import_ok":     {"en": "Model loaded successfully.",
                                 "hu": "Modell sikeresen betöltve."},
    "deep_model_import_err":    {"en": "Could not load the model file. "
                                 "It may be corrupt or from an incompatible version.",
                                 "hu": "A modellfájl nem tölthető be. "
                                 "Sérült vagy inkompatibilis verzióból származhat."},
    "deep_model_title":         {"en": "AI Model", "hu": "AI modell"},

}
