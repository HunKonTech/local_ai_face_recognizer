"""UI strings (EN + HU) — part 6, starting at: Face quality filter settings."""

from __future__ import annotations

from typing import Dict

STRINGS: Dict[str, Dict[str, str]] = {
    # ── Face quality filter settings ──────────────────────────────────────
    "fq_group":             {"en": "Face Quality Filter",
                             "hu": "Arc minőségszűrő"},
    "fq_exclude_toggle":    {"en": "Exclude low-quality faces from model building",
                             "hu": "Gyenge minőségű arcok kizárása modellépítésből"},
    "fq_exclude_tip":       {"en": "When enabled, back-facing, blurry, too-small or low-confidence "
                                   "face detections are skipped during embedding, recognition, "
                                   "clustering and name suggestions. "
                                   "The bounding box is still shown and manual name assignment "
                                   "always works regardless of this setting.",
                             "hu": "Ha be van kapcsolva, a háttal álló, homályos, túl kicsi vagy "
                                   "alacsony konfidenciájú arcdetektálások ki lesznek hagyva az "
                                   "embedding, felismerés, klaszterezés és névajánlatok során. "
                                   "A bounding box továbbra is látható marad, és a kézi "
                                   "névhozzárendelés mindig működik függetlenül ettől a beállítástól."},
    "fq_thresholds_group":  {"en": "Quality Thresholds",
                             "hu": "Minőségi küszöbértékek"},
    "fq_min_confidence":    {"en": "Min. detection confidence:",
                             "hu": "Min. detektálási konfidencia:"},
    "fq_min_area":          {"en": "Min. face area (px²):",
                             "hu": "Min. arcterület (px²):"},
    "fq_min_sharpness":     {"en": "Min. sharpness (Laplacian variance):",
                             "hu": "Min. élességi érték (Laplacian variancia):"},
    "fq_reanalyze_btn":     {"en": "Re-evaluate All Faces …",
                             "hu": "Összes arc újraértékelése …"},
    "fq_reanalyze_confirm": {"en": "Re-evaluate quality for all {n} face(s) in the database?\n"
                                   "This may take a few minutes.",
                             "hu": "Újraértékeli a minőséget az adatbázis mind a(z) {n} arcán?\n"
                                   "Ez néhány percig tarthat."},
    "fq_reanalyze_done":    {"en": "Quality re-evaluation complete: {n} face(s) processed.",
                             "hu": "Minőségi újraértékelés kész: {n} arc feldolgozva."},
    "fq_low_quality_tip":   {"en": "Low-quality face — excluded from model building\n"
                                   "(back-facing / blurry / too small / low confidence)\n"
                                   "Right-click to override manually.",
                             "hu": "Gyenge minőségű arc — kizárva a modellépítésből\n"
                                   "(háttal álló / homályos / túl kicsi / alacsony konfidencia)\n"
                                   "Jobb klikk a kézi felülbíráláshoz."},
    "fq_ctx_force_include":  {"en": "Force include in model building",
                              "hu": "Kényszerített belefoglalás a modellépítésbe"},
    "fq_ctx_force_exclude":  {"en": "Exclude from model building",
                              "hu": "Kizárás a modellépítésből"},

    # ── Deoldified pairing ────────────────────────────────────────────────
    "geocoding_group":          {"en": "Address geocoding", "hu": "Cím-geokódolás"},
    "geocoding_enable":         {"en": "Enable online geocoding (Nominatim / OpenStreetMap)",
                                 "hu": "Online geokódolás engedélyezése (Nominatim / OpenStreetMap)"},
    "geocoding_enable_tip":     {"en": "Off by default. When enabled, settlement/street autocomplete "
                                       "and address lookups query OpenStreetMap. Results are cached "
                                       "locally; your own entered addresses always work offline.",
                                 "hu": "Alapból kikapcsolva. Bekapcsolva a település/utca kiegészítés és "
                                       "a címkeresés az OpenStreetMap-et kérdezi. Az eredmények helyben "
                                       "cache-elődnek; a saját beírt címek offline is működnek."},
    "deoldified_group":         {"en": "Deoldified / Colorized Pairing",
                                 "hu": "Deoldified / Színezett képpárosítás"},
    "deoldified_toggle":        {"en": "Share data between paired images "
                                       "(faces, image data, object tags)",
                                 "hu": "Adatok megosztása a párosított képek közt "
                                       "(arcok, képadatok, objektumjelölések)"},
    "deoldified_toggle_tip":    {"en": "When enabled, an image containing '-deoldified' in its "
                                       "filename shares its annotations with the original "
                                       "black-and-white photo: face data from the original is "
                                       "shown on both versions, and object tags are kept on the "
                                       "original so the photo is counted once.",
                                 "hu": "Ha be van kapcsolva, a '-deoldified' szót tartalmazó kép "
                                       "megosztja a jelöléseit az eredeti fekete-fehér képpel: "
                                       "az arcadatok az eredetiről mindkét változaton "
                                       "megjelennek, az objektumjelölések pedig az eredetin "
                                       "maradnak, így a fénykép egyszer számít."},
    "deoldified_view_always":   {"en": "The black-and-white / colorized view switcher always "
                                       "appears above the image when a pair exists; it does not "
                                       "depend on this option.",
                                 "hu": "A fekete-fehér / színezett nézetváltó mindig megjelenik "
                                       "a kép felett, ha van párja; ez a beállítás nem "
                                       "befolyásolja."},
    "deoldified_sync_toggle":   {"en": "Automatically copy data between paired images",
                                 "hu": "Adatok automatikus átvétele a párképek közt"},
    "deoldified_sync_toggle_tip": {"en": "When a deoldified pair is opened, copy whatever the "
                                       "colorized side is missing: faces, image data and object "
                                       "tags. Existing data is never overwritten.",
                                 "hu": "Deoldified pár megnyitásakor mindannak az átmásolása, "
                                       "ami a színezett oldalról hiányzik: arcok, képadatok és "
                                       "objektumjelölések. Meglévő adatot soha nem ír felül."},
    "ibp_deol_pair_lbl":        {"en": "View:",               "hu": "Nézet:"},
    "ibp_view_original_bw":     {"en": "Original B&W",        "hu": "Eredeti fekete-fehér"},
    "ibp_view_colorized":       {"en": "Colorized",            "hu": "Színezett változat"},
    "ibp_view_compare":         {"en": "Compare",              "hu": "Összehasonlítás"},
    "ibp_view_compare_tip":     {"en": "Drag the slider to reveal the colorized image on the "
                                       "right and the black-and-white on the left.",
                                 "hu": "Húzd a csúszkát: jobbra a színezett, balra a "
                                       "fekete-fehér változat látszik."},
    "ibp_deol_left":            {"en": "Left:",                "hu": "Bal:"},
    "ibp_deol_right":           {"en": "Right:",               "hu": "Jobb:"},
    "ibp_deol_variant":         {"en": "Variant:",             "hu": "Változat:"},
    "ibp_deol_bw_label":        {"en": "Black & white",        "hu": "Fekete-fehér"},
    "ibp_deol_variant_missing": {"en": "Colorized version file is missing; showing black & white.",
                                 "hu": "A színezett változat fájlja hiányzik; a fekete-fehér nézet látható."},
    "ibp_deol_side_missing":    {"en": "This version's file cannot be opened; it may have been "
                                       "moved or the image library root is wrong.",
                                 "hu": "Ennek a változatnak a fájlja nem nyitható meg; lehet, hogy "
                                       "elmozdult, vagy rossz a képkönyvtár gyökere."},
    "ibp_deol_fix_paths":       {"en": "Fix paths…",           "hu": "Útvonalak javítása…"},
    "ibp_deol_fix_paths_tip":   {"en": "Look for the missing files and re-point the database at "
                                       "them.",
                                 "hu": "A hiányzó fájlok megkeresése, és az adatbázis "
                                       "hozzáigazítása."},
    "ibp_deol_sync":            {"en": "Copy data from pair",  "hu": "Adatok átvétele a párról"},
    "ibp_deol_sync_tip":        {"en": "Copy whatever the paired image is missing: faces, image "
                                       "data and object tags. Existing data is never overwritten.",
                                 "hu": "Minden átmásolása, ami a párképről hiányzik: arcok, "
                                       "képadatok és objektumjelölések. Meglévő adatot soha nem "
                                       "ír felül."},
    "ibp_deol_sync_done":       {"en": "Copied {faces} new face(s), completed {updated} face(s), "
                                       "moved {objects} object tag(s) and copied {fields} "
                                       "metadata field(s).",
                                 "hu": "{faces} új arc átmásolva, {updated} arc kiegészítve, "
                                       "{objects} objektumjelölés áthelyezve és {fields} "
                                       "metaadat-mező átmásolva."},
    "ibp_deol_sync_skipped":    {"en": "Nothing new to copy: the pair is already in sync.",
                                 "hu": "Nincs átmásolható új adat: a pár már szinkronban van."},
    "ibp_deol_sync_no_pair":    {"en": "No paired image found for the current image.",
                                 "hu": "Nincs párkép a jelenlegi képhez."},
    "ibp_deol_settings_tip":    {"en": "Pairing settings — decide whether the two versions share "
                                       "faces, image data and object tags.",
                                 "hu": "Párosítási beállítások — itt döntheted el, hogy a két "
                                       "változat megosztja-e az arcokat, a képadatokat és az "
                                       "objektumjelöléseket."},

    # ── Image library ─────────────────────────────────────────────────────
    "ibp_deol_variant_type":    {"en": "Type:",                  "hu": "Típus:"},
    "ibp_deol_variant_artistic":{"en": "(artistic)",             "hu": "(szövegezetesen szinezett)"},
    "ibp_deol_variant_stable":  {"en": "(stable)",               "hu": "(stabilizált)"},

    "img_lib_group":            {"en": "Image Library",
                                 "hu": "Képkönyvtár"},
    "img_lib_root_label":       {"en": "Library root:",
                                 "hu": "Könyvtár gyökere:"},
    "img_lib_not_configured":   {"en": "(not configured — using absolute paths)",
                                 "hu": "(nincs beállítva — abszolút útvonalakat használ)"},
    "img_lib_status_ok":        {"en": "Connected: {path}",
                                 "hu": "Csatlakoztatva: {path}"},
    "img_lib_status_missing":   {"en": "Root not found: {path}",
                                 "hu": "Gyökér nem található: {path}"},
    "img_lib_change_btn":       {"en": "Change …",
                                 "hu": "Módosítás …"},
    "img_lib_migrate_btn":      {"en": "Migrate to Relative Paths",
                                 "hu": "Relatív útvonalakra konvertálás"},
    "img_lib_migrate_tip":      {"en": "Convert all absolute image paths in the database "
                                       "to relative paths based on the library root. "
                                       "Run this once after setting the root for the first time.",
                                 "hu": "Az adatbázisban tárolt abszolút útvonalak konvertálása "
                                       "relatív útvonalakra a könyvtár gyökere alapján. "
                                       "Ezt egyszer kell futtatni a gyökér első beállítása után."},
    "img_lib_select_title":     {"en": "Select Image Library Root Folder",
                                 "hu": "Képkönyvtár gyökérmappájának kiválasztása"},
    "img_lib_migrate_title":    {"en": "Migrate Image Paths",
                                 "hu": "Képútvonalak migrálása"},
    "img_lib_migrate_confirm":  {"en": "Convert {n} image record(s) to relative paths?\n\n"
                                       "Library root: {root}\n\n"
                                       "The original file_path values are preserved. "
                                       "The migration can be repeated safely.",
                                 "hu": "Konvertálja a(z) {n} képrekordot relatív útvonalakra?\n\n"
                                       "Könyvtár gyökere: {root}\n\n"
                                       "Az eredeti fájlútvonalak megmaradnak. "
                                       "A migráció biztonságosan ismételhető."},
    "img_lib_migrating":        {"en": "Migrating image paths …",
                                 "hu": "Képútvonalak migrálása …"},
    "img_lib_migrate_done":     {"en": "Migration complete: {migrated} converted, {skipped} skipped.",
                                 "hu": "Migráció kész: {migrated} konvertálva, {skipped} kihagyva."},
    "img_lib_migrate_errors":   {"en": "Files not found during migration ({n}):",
                                 "hu": "Migráció során nem talált fájlok ({n}):"},
    "img_lib_missing_title":    {"en": "Image Library Not Found",
                                 "hu": "Képkönyvtár nem található"},
    "img_lib_missing_msg":      {"en": "The configured image library root was not found:\n\n"
                                       "{path}\n\n"
                                       "Images cannot be loaded until a valid root is selected. "
                                       "Choose the new location of your image folder, or skip to "
                                       "continue with reduced functionality.",
                                 "hu": "A beállított képkönyvtár gyökere nem található:\n\n"
                                       "{path}\n\n"
                                       "A képek nem tölthetők be érvényes gyökér nélkül. "
                                       "Válassza ki a képkönyvtár új helyét, vagy ugorja át "
                                       "a korlátozott működés folytatásához."},
    "img_lib_find_btn":         {"en": "Find New Location …",
                                 "hu": "Új helyszín keresése …"},
    "img_lib_skip_btn":         {"en": "Skip for Now",
                                 "hu": "Kihagyás egyelőre"},
    "img_lib_root_changed":     {"en": "Image library root updated.",
                                 "hu": "Képkönyvtár gyökere frissítve."},
    "img_lib_root_no_match":    {"en": "None of {checked} checked images can be found under "
                                       "{path}. The stored relative paths belong to a different "
                                       "root, so every image would break. Use this folder anyway?",
                                 "hu": "A megvizsgált {checked} képből egyet sem találtam a "
                                       "{path} mappa alatt. A tárolt relatív útvonalak más "
                                       "gyökérhez tartoznak, így minden kép elérhetetlenné "
                                       "válna. Mégis ezt a mappát választod?"},
    "img_lib_n_relative":       {"en": "{n} image(s) with relative paths",
                                 "hu": "{n} kép relatív útvonallal"},
    "img_lib_n_absolute":       {"en": "{n} image(s) with absolute paths only",
                                 "hu": "{n} kép csak abszolút útvonallal"},
    "img_lib_no_root_for_migrate": {"en": "Please set the library root before migrating.",
                                    "hu": "A migráció előtt állítsa be a könyvtár gyökerét."},

    # ── Scan modes dialog ─────────────────────────────────────────────────
    "scanModes.title":
        {"en": "Scan & Maintenance",
         "hu": "Beolvasás és karbantartás"},
    # ── Deep-learning (AI) operations ─────────────────────────────────────
    "scanModes.deep.intro":
        {"en": "The new deep-learning engine trains a neural network from the people "
               "you have already categorized, then places unknown faces with them. "
               "The more faces a person has, the more accurate it gets — it learns "
               "from every run and from every correction you make. Training favours "
               "accuracy over speed and may fully load the CPU for a while.",
         "hu": "Az új mélytanulásos motor a már kategorizált személyekből neurális "
               "hálót tanít, majd az ismeretlen arcokat próbálja hozzájuk rendelni. "
               "Minél több arc tartozik egy személyhez, annál pontosabb — minden "
               "futtatásból és minden javításodból tanul. A tanítás a pontosságot "
               "részesíti előnyben a sebességgel szemben, és egy ideig teljesen "
               "leterhelheti a processzort."},
    "scanModes.deepRescan.title":
        {"en": "AI Re-scan",
         "hu": "AI újra beolvasás"},
    "scanModes.deepRescan.description":
        {"en": "Scans for new images, then retrains the neural network from your "
               "categorized faces and tries to place every uncategorized, unknown "
               "face with a known person. Overlapping duplicate boxes of the same "
               "face are cleaned up: a box already assigned to a person always "
               "survives; between two unknown boxes one is kept. Faces already "
               "recognized are never touched — the earlier recognition wins.",
         "hu": "Beolvassa az új képeket, majd a kategorizált arcokból újratanítja a "
               "neurális hálót, és minden nem kategorizált, ismeretlen arcot "
               "megpróbál egy ismert személyhez rendelni. Az ugyanarra az arcra "
               "rajzolt átfedő, dupla kereteket kitisztítja: a már személyhez "
               "rendelt keret mindig megmarad; két ismeretlen keretből egy marad. "
               "A már felismert arcokhoz nem nyúl — a korábbi felismerés marad."},
    "scanModes.deepRescan.startButton":
        {"en": "Start AI Re-scan",
         "hu": "AI újra beolvasás indítása"},
    "scanModes.deepRescan.warning":
        {"en": "Safe — existing assignments are never changed; training can take a while",
         "hu": "Biztonságos — a meglévő hozzárendelések nem változnak; a tanítás eltarthat egy ideig"},
    "scanModes.deepTrain.title":
        {"en": "Train the Model",
         "hu": "Modell tanítása"},
    "scanModes.deepTrain.description":
        {"en": "Walks through every face already recognized and assigned to a "
               "person and retrains the neural network from them. Nothing else "
               "happens: no new scan, no recognition, not a single face is "
               "moved or changed. Run it after you have confirmed or corrected "
               "many faces, so the model catches up with what it learned from "
               "you before the next re-scan.",
         "hu": "Végigmegy az összes már felismert és személyhez rendelt arcon, "
               "és ezekből újratanítja a neurális hálót. Más nem történik: "
               "nincs új beolvasás, nincs felismerés, egyetlen arc sem mozdul "
               "vagy változik. Akkor érdemes futtatni, ha sok arcot "
               "megerősítettél vagy javítottál — így a modell már a következő "
               "újra beolvasás előtt naprakész lesz a tőled tanultakkal."},
    "scanModes.deepTrain.startButton":
        {"en": "Train Model",
         "hu": "Modell tanítása"},
    "scanModes.deepTrain.warning":
        {"en": "Safe — your data is not modified at all; training can take a "
               "while and may use 100% CPU",
         "hu": "Biztonságos — az adataid egyáltalán nem módosulnak; a tanítás "
               "eltarthat egy ideig és 100%-on járathatja a processzort"},
    "scanModes.deepFaceDetect.title":
        {"en": "AI Face Detection",
         "hu": "AI arc detektálás"},
    "scanModes.deepFaceDetect.description":
        {"en": "Analysis only, powered by a pretrained deep-learning detector: "
               "checks every image for whether it contains faces, how many, "
               "where they are (bounding boxes) and with what confidence. The "
               "results are saved and can be looked up later. No identification "
               "happens — no face is assigned, renamed, moved or deleted, and "
               "the classic recognition results are not touched at all.",
         "hu": "Csak elemzés, előtanított mélytanulásos detektorral: minden "
               "képen megvizsgálja, van-e rajta arc, hány darab, hol "
               "helyezkednek el (befoglaló keretek) és mekkora bizonyossággal. "
               "Az eredmények mentésre kerülnek, később visszakereshetők. "
               "Azonosítás nem történik — egyetlen arc sem kerül "
               "hozzárendelésre, átnevezésre, áthelyezésre vagy törlésre, és a "
               "klasszikus felismerés eredményeihez sem nyúl."},
    "scanModes.deepFaceDetect.startButton":
        {"en": "Start AI Face Detection",
         "hu": "AI arc detektálás indítása"},
    "scanModes.deepFaceDetect.warning":
        {"en": "Safe — analysis only: nothing is assigned, modified or deleted",
         "hu": "Biztonságos — csak elemzés: semmi sem kerül hozzárendelésre, "
               "módosításra vagy törlésre"},
    "scanModes.deepRebuild.title":
        {"en": "AI Rebuild from Scratch",
         "hu": "AI újraépítés nulláról"},
    "scanModes.deepRebuild.description":
        {"en": "Rebuilds the face database from zero: every automatically detected "
               "face box is deleted, all images are re-detected and re-embedded, "
               "then the neural network is trained and recognition runs on the "
               "fresh data. Manually drawn boxes and the faces you confirmed on "
               "people are kept — they are the training data the network learns "
               "from. Slow on large libraries, but gives the cleanest result.",
         "hu": "Nulláról újraépíti az arc-adatbázist: minden automatikusan felismert "
               "arckeretet töröl, minden képet újra feldolgoz (detektálás + "
               "embedding), majd a friss adatokon tanítja a neurális hálót és "
               "futtatja a felismerést. A kézzel rajzolt keretek és a személyeknél "
               "megerősített arcok megmaradnak — ezekből tanul a háló. Nagy "
               "könyvtárnál lassú, de ez adja a legtisztább eredményt."},
    "scanModes.deepRebuild.startButton":
        {"en": "Rebuild from Scratch",
         "hu": "Újraépítés nulláról"},
    "scanModes.deepRebuild.warning":
        {"en": "Destructive — automatic face boxes and Unknown groups are rebuilt; "
               "human-confirmed faces survive",
         "hu": "Romboló — az automatikus arckeretek és Unknown csoportok újraépülnek; "
               "az ember által megerősített arcok megmaradnak"},
    "deep_rebuild_confirm_title":
        {"en": "AI Rebuild from Scratch",
         "hu": "AI újraépítés nulláról"},
    "deep_rebuild_confirm_msg":
        {"en": "Rebuild the face database from zero?\n\n"
               "All automatically detected face boxes and Unknown groups will be "
               "deleted and recreated. Manually drawn boxes and human-confirmed "
               "person assignments are kept as training data.\n\n"
               "This can take a long time on a large library.",
         "hu": "Nulláról újraépíti az arc-adatbázist?\n\n"
               "Minden automatikusan felismert arckeret és Unknown csoport törlődik, "
               "majd újraépül. A kézzel rajzolt keretek és az ember által "
               "megerősített személy-hozzárendelések tanítóadatként megmaradnak.\n\n"
               "Nagy képkönyvtárnál ez sokáig tarthat."},

    # ── Force-rebuild the neural model only (no scan/detection) ────────────
    "scanModes.rebuildModel.title":
        {"en": "Force-Rebuild AI Model",
         "hu": "AI modell kényszerített újraépítése"},
    "scanModes.rebuildModel.description":
        {"en": "Deletes the current neural model and trains a brand-new one from "
               "scratch on your labeled faces, then re-recognizes the unknown "
               "faces. Face boxes and detections are NOT touched — only the model "
               "is rebuilt, so it is much faster than a full rebuild. Use this "
               "after resetting the database, or whenever the model seems off and "
               "you want a guaranteed fresh model (never reuses the old one).",
         "hu": "Törli a jelenlegi neurális modellt, és teljesen nulláról épít egy "
               "újat a felcímkézett arcokból, majd újra felismeri az ismeretlen "
               "arcokat. Az arckereteket és a detektálást NEM érinti — csak a "
               "modellt építi újra, így sokkal gyorsabb a teljes újraépítésnél. "
               "Használd adatbázis-reset után, vagy ha a modell gyanúsan "
               "viselkedik és garantáltan friss modellt szeretnél (sosem "
               "használja újra a régit)."},
    "scanModes.rebuildModel.startButton":
        {"en": "Rebuild Model from Scratch",
         "hu": "Modell újraépítése nulláról"},
    "scanModes.rebuildModel.warning":
        {"en": "Replaces the AI model — face boxes are kept, only the model is rebuilt",
         "hu": "Lecseréli az AI modellt — az arckeretek megmaradnak, csak a modell épül újra"},
    "rebuild_model_confirm_title":
        {"en": "Force-Rebuild AI Model",
         "hu": "AI modell kényszerített újraépítése"},
    "rebuild_model_confirm_msg":
        {"en": "Delete the current AI model and train a new one from scratch?\n\n"
               "Your face boxes and person assignments are kept; only the neural "
               "model is rebuilt and unknown faces are re-recognized.",
         "hu": "Törli a jelenlegi AI modellt, és teljesen nulláról tanít egy újat?\n\n"
               "Az arckeretek és a személy-hozzárendelések megmaradnak; csak a "
               "neurális modell épül újra, és az ismeretlen arcokat újra felismeri."},

    # ── Automatic groupings (deep recognition review) ─────────────────────
    "auto_assignments_found_title":
        {"en": "AI Groupings Ready",
         "hu": "AI csoportosítások elkészültek"},
    "auto_assignments_found_msg":
        {"en": "The AI placed {n} face(s) with known people in this run.\n\n"
               "Would you like to review them now? You can confirm, correct or "
               "undo each grouping — the AI learns from your corrections.",
         "hu": "Az AI ebben a futásban {n} arcot rendelt ismert személyekhez.\n\n"
               "Átnézed most? Mindegyik csoportosítást megerősítheted, javíthatod "
               "vagy visszavonhatod — az AI tanul a javításaidból."},
    "autoAssign.intro":
        {"en": "Faces the last AI run placed with known people. Confirm the good "
               "ones, fix the wrong ones — the network learns from every decision "
               "you make here.",
         "hu": "Arcok, amelyeket a legutóbbi AI futás ismert személyekhez rendelt. "
               "Erősítsd meg a jókat, javítsd a rosszakat — a háló minden itteni "
               "döntésedből tanul."},
    "autoAssign.empty":
        {"en": "No automatic groupings to review. Run an AI re-scan from "
               "Scan & Maintenance to get new ones.",
         "hu": "Nincs átnézendő automatikus csoportosítás. Futtass AI újra "
               "beolvasást a Beolvasás és karbantartás ablakból."},
    "autoAssign.run_info":
        {"en": "Last AI run: {date}",
         "hu": "Utolsó AI futás: {date}"},
    "autoAssign.count":
        {"en": "{n} grouping(s) to review",
         "hu": "{n} átnézendő csoportosítás"},
    "autoAssign.from_group":
        {"en": "previously: {name}",
         "hu": "korábban: {name}"},
    "autoAssign.from_unassigned":
        {"en": "previously: unassigned",
         "hu": "korábban: hozzárendelés nélkül"},
    "autoAssign.confirm":
        {"en": "Correct",
         "hu": "Jó"},
    "autoAssign.confirm_tooltip":
        {"en": "Confirm the grouping — the face becomes a trusted training example",
         "hu": "Megerősítés — az arc megbízható tanítópéldává válik"},
    "autoAssign.correct":
        {"en": "Change Person…",
         "hu": "Másik személy…"},
    "autoAssign.correct_tooltip":
        {"en": "Move the face to the right person — the AI learns from the correction",
         "hu": "Az arc áthelyezése a helyes személyhez — az AI tanul a javításból"},
    "autoAssign.revert":
        {"en": "Undo",
         "hu": "Visszavonás"},
    "autoAssign.revert_tooltip":
        {"en": "Send the face back to where it was before the run",
         "hu": "Az arc visszakerül oda, ahol a futás előtt volt"},
    "autoAssign.select_all":
        {"en": "Select all",
         "hu": "Összes kijelölése"},
    "autoAssign.select_all_tooltip":
        {"en": "Tick every grouping in the review list",
         "hu": "Minden csoportosítás kijelölése a listában"},
    "autoAssign.clear_selection":
        {"en": "Clear selection",
         "hu": "Kijelölés törlése"},
    "autoAssign.clear_selection_tooltip":
        {"en": "Untick every grouping",
         "hu": "Minden kijelölés megszüntetése"},
    "autoAssign.accept_selected":
        {"en": "Accept selected ({n})",
         "hu": "Kijelöltek elfogadása ({n})"},
    "autoAssign.accept_selected_tooltip":
        {"en": "Confirm every ticked grouping at once — each becomes trusted "
               "training data and moves to the reviewed list",
         "hu": "Minden kijelölt csoportosítás egyszerre megerősítve — mind "
               "megbízható tanítópéldává válik és átkerül az átnézett listába"},
    "autoAssign.threshold_label":
        {"en": "Match ≥",
         "hu": "Egyezés ≥"},
    "autoAssign.select_above":
        {"en": "Select ≥ {pct}% ({n})",
         "hu": "{pct}% felettiek kijelölése ({n})"},
    "autoAssign.select_above_tooltip":
        {"en": "Tick exactly the groupings whose match is at or above the "
               "threshold, so you can look them over before accepting",
         "hu": "Pontosan azok a csoportosítások lesznek kijelölve, amelyek "
               "egyezése eléri a küszöböt — elfogadás előtt átnézheted őket"},
    "autoAssign.accept_above":
        {"en": "Accept all ≥ {pct}% ({n})",
         "hu": "Minden elfogadása {pct}% felett ({n})"},
    "autoAssign.accept_above_tooltip":
        {"en": "Confirm every grouping whose match is at or above the chosen "
               "percentage — each becomes trusted training data",
         "hu": "Minden csoportosítás megerősítése, amelynek egyezése eléri a "
               "megadott százalékot — mind megbízható tanítópéldává válik"},
    "autoAssign.accept_above_confirm_title":
        {"en": "Accept groupings above the threshold?",
         "hu": "Küszöb feletti csoportosítások elfogadása?"},
    "autoAssign.accept_above_confirm_msg":
        {"en": "{n} automatic groupings with a match of {pct}% or higher will "
               "be confirmed. The AI treats them as trusted training data.",
         "hu": "{n} automatikus csoportosítás, amelynek egyezése legalább "
               "{pct}%, megerősítésre kerül. Az AI ezeket megbízható "
               "tanítópéldaként kezeli."},
    "autoAssign.reject_selected":
        {"en": "Reject selected ({n})",
         "hu": "Kijelöltek elutasítása ({n})"},
    "autoAssign.reject_selected_tooltip":
        {"en": "Undo every ticked grouping at once — each face returns to where "
               "it was before the run and the AI learns from the rejection",
         "hu": "Minden kijelölt csoportosítás egyszerre visszavonva — minden arc "
               "visszakerül a futás előtti helyére, és az AI tanul az elutasításból"},
    "autoAssign.assign_selected":
        {"en": "Move to person… ({n})",
         "hu": "Másik személyhez… ({n})"},
    "autoAssign.assign_selected_tooltip":
        {"en": "Move every ticked face to one chosen (or new) person at once — "
               "the AI learns from the correction",
         "hu": "Minden kijelölt arc áthelyezése egy kiválasztott (vagy új) "
               "személyhez egyszerre — az AI tanul a javításból"},
    "autoAssign.override":
        {"en": "Override",
         "hu": "Felülbírálás"},
    "autoAssign.override_tooltip":
        {"en": "Change your mind: undo this decision and send the face back to "
               "where it was before the run",
         "hu": "Döntés felülbírálása: visszavonás, az arc visszakerül oda, ahol "
               "a futás előtt volt"},
    "autoAssign.graph":
        {"en": "Decision graph — why the AI suggested this",
         "hu": "Döntési gráf — mi alapján javasolta az AI"},
    "autoAssign.graph_title":
        {"en": "AI decision graph",
         "hu": "AI döntési gráf"},
    "autoAssign.graph_tab_flow":
        {"en": "Decision flow",
         "hu": "Döntési folyamat"},
    "autoAssign.graph_tab_nn":
        {"en": "Neural network",
         "hu": "Neurális háló"},
    "autoAssign.graph_nn_legend":
        {"en": "Green: the neurons that fired the strongest and the path "
               "through which their signal reached the final decision "
               "(the winning person). Orange intensity shows the raw "
               "activation of each neuron.",
         "hu": "Zöld: a legerősebben aktiválódott neuronok és az az út, "
               "amelyen a jelük a végső döntésig (a nyertes személyig) "
               "eljutott. A narancs árnyalat az egyes neuronok nyers "
               "aktivációját mutatja."},
    "autoAssign.graph_nn_unavailable":
        {"en": "The neural-network view cannot be computed: no trained model, "
               "no stored embedding for this face, or the model runs in "
               "prototype mode (too few people/examples for the network).",
         "hu": "A neurális háló nézet nem számítható ki: nincs betanított "
               "modell, nincs mentett beágyazás ehhez az archoz, vagy a "
               "modell prototípus módban fut (túl kevés személy/példa a "
               "hálóhoz)."},
    "autoAssign.revert_all":
        {"en": "Undo all",
         "hu": "Mind visszavonása"},
    "autoAssign.revert_all_tooltip":
        {"en": "Send every unreviewed face of this run back to where it was",
         "hu": "A futás minden még nem ellenőrzött arca visszakerül oda, ahol volt"},
    "autoAssign.revert_all_confirm_title":
        {"en": "Undo all groupings?",
         "hu": "Minden csoportosítás visszavonása?"},
    "autoAssign.revert_all_confirm_msg":
        {"en": "All {n} unreviewed automatic groupings of this run will be "
               "undone — every face returns to where it was before the run. "
               "The AI records these as mistakes and learns from them. "
               "Already confirmed or corrected faces are not touched.",
         "hu": "A futás mind a(z) {n} még nem ellenőrzött automatikus "
               "csoportosítása visszavonásra kerül — minden arc visszakerül "
               "oda, ahol a futás előtt volt. Az AI ezeket hibaként jegyzi "
               "meg és tanul belőlük. A már megerősített vagy javított "
               "arcokhoz nem nyúl."},

}
