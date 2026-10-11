"""UI strings (EN + HU) — part 3, starting at: AI model tab (Settings)."""

from __future__ import annotations

from typing import Dict

STRINGS: Dict[str, Dict[str, str]] = {
    # ── AI model tab (Settings) ───────────────────────────────────────────
    "settings_tab_ai_model":    {"en": "AI Model", "hu": "AI modell"},
    "ai_model_layers_group":    {"en": "Neural network architecture",
                                 "hu": "Neurális háló felépítése"},
    "ai_model_layers_desc":     {"en": "The face recognizer is an ensemble of neural "
                                 "networks. Each network receives the 192-dimensional "
                                 "face embedding as input, passes it through the hidden "
                                 "layers configured below, and a final output layer "
                                 "(the actual inference) produces a probability for "
                                 "every known person. More / larger hidden layers give "
                                 "the model more capacity — useful with many people and "
                                 "many confirmed faces — but training becomes slower "
                                 "and, with little training data, the model can overfit "
                                 "(it memorizes the examples instead of generalizing). "
                                 "With few labeled faces, fewer / smaller layers are "
                                 "usually more accurate.",
                                 "hu": "Az arcfelismerő több neurális hálóból álló "
                                 "együttes. Minden háló bemenete a 192 dimenziós "
                                 "arc-beágyazás, ezt az alább beállított rejtett "
                                 "rétegeken vezeti át, majd egy záró kimeneti réteg "
                                 "(a tényleges következtetés) ad valószínűséget minden "
                                 "ismert személyre. Több / nagyobb rejtett réteg "
                                 "nagyobb kapacitást ad — sok személynél és sok "
                                 "megerősített arcnál hasznos —, viszont a tanítás "
                                 "lassabb lesz, és kevés tanítóadatnál a modell "
                                 "túltanulhat (bemagolja a példákat ahelyett, hogy "
                                 "általánosítana). Kevés címkézett arcnál általában a "
                                 "kevesebb / kisebb réteg pontosabb."},
    "ai_model_layer_count":     {"en": "Number of hidden layers:",
                                 "hu": "Rejtett rétegek száma:"},
    "ai_model_layer_n":         {"en": "Layer {n} size (neurons):",
                                 "hu": "{n}. réteg mérete (neuron):"},
    "ai_model_reset_btn":       {"en": "Restore defaults (256, 192, 128, 64)",
                                 "hu": "Alapértelmezés visszaállítása (256, 192, 128, 64)"},
    "ai_model_rebuild_note":    {"en": "Changes take effect after the next AI model "
                                 "training (AI rescan or model rebuild). The layer "
                                 "sizes usually taper from input towards the output.",
                                 "hu": "A módosítás a következő AI-modell tanításnál "
                                 "lép életbe (AI beolvasás vagy modell-újraépítés). "
                                 "A rétegméretek jellemzően a bemenettől a kimenet "
                                 "felé csökkennek."},

    # ── Debug tab (Settings) ──────────────────────────────────────────────
    "settings_tab_debug":       {"en": "Debug", "hu": "Debug"},
    "debug_ai_viz_group":       {"en": "AI Visualization", "hu": "AI vizualizáció"},
    "debug_ai_viz_check":       {"en": "Show AI decision window during scan",
                                 "hu": "AI döntési ablak megjelenítése beolvasás közben"},
    "debug_ai_viz_desc":        {"en": "Opens a live window showing the face crop, "
                                 "neuron activations and output probabilities for each "
                                 "face as the AI processes it.",
                                 "hu": "Valós idejű ablakot nyit, amely minden feldolgozott "
                                 "arcnál mutatja a kivágást, a neuron-aktivációkat és a "
                                 "kimeneti valószínűségeket."},
    "debug_log_group":          {"en": "Decision Log", "hu": "Döntési napló"},
    "debug_log_check":          {"en": "Write detailed decision log (deep_debug.jsonl)",
                                 "hu": "Részletes döntési napló írása (deep_debug.jsonl)"},
    "debug_log_desc":           {"en": "Appends one JSON line per face to "
                                 "data/deep_debug.jsonl — contains gate results, "
                                 "similarity scores and the final decision. "
                                 "Useful for reverse-engineering why a face was accepted "
                                 "or rejected.",
                                 "hu": "Minden archoz egy JSON sort fűz a "
                                 "data/deep_debug.jsonl fájlhoz — tartalmazza a gate "
                                 "eredményeket, hasonlóság-értékeket és a végső döntést. "
                                 "Hasznos annak kiderítésére, miért lett elfogadva vagy "
                                 "elutasítva egy arc."},
    "debug_log_open_btn":       {"en": "Open log file", "hu": "Naplófájl megnyitása"},
    "debug_log_clear_btn":      {"en": "Clear log", "hu": "Napló törlése"},
    "debug_log_cleared":        {"en": "Debug log cleared.", "hu": "Debug napló törölve."},
    "debug_log_not_found":      {"en": "No debug log found yet.", "hu": "Még nincs debug napló."},

    # ── Detection debug log ───────────────────────────────────────────────
    "debug_detection_log_group": {
        "en": "Detection Debug Log",
        "hu": "Detektálás Debug Log",
    },
    "debug_detection_log_desc": {
        "en": (
            "Writes a timestamped log file for each detection run. "
            "Every image is listed with all raw bounding boxes and which ones "
            "were kept or dropped by the verification filter. "
            "Useful for diagnosing platform-specific false positives."
        ),
        "hu": (
            "Minden detektálási futáshoz létrehoz egy időbélyeges log fájlt. "
            "Minden képnél látható az összes nyers bounding box, és hogy melyeket "
            "tartotta meg vagy dobta el a verifier szűrő. "
            "Hasznos a platformspecifikus fals pozitívok vizsgálatához."
        ),
    },
    "debug_detection_log_check": {
        "en": "Save detection details to log file per run",
        "hu": "Detektálás részleteinek mentése log fájlba futásonként",
    },
    "debug_detection_log_open_btn": {
        "en": "Open log folder",
        "hu": "Log mappa megnyitása",
    },
    "debug_detection_log_folder_empty": {
        "en": "No detection log files found yet.\nEnable the option and run a scan.",
        "hu": "Még nincs detektálás log fájl.\nKapcsold be a kapcsolót és futtass egy szkennelést.",
    },

    # ── AI visualization window ───────────────────────────────────────────
    "debug_viz_title":          {"en": "AI Decision Visualizer", "hu": "AI döntés vizualizátor"},
    "debug_viz_gates":          {"en": "Gates", "hu": "Kapuk"},
    "debug_viz_layer":          {"en": "Layer", "hu": "Réteg"},
    "debug_viz_neurons":        {"en": "neurons", "hu": "neuron"},
    "debug_viz_output_probs":   {"en": "Output probabilities", "hu": "Kimeneti valószínűségek"},
    "debug_viz_person":         {"en": "Person", "hu": "Személy"},
    "debug_viz_score":          {"en": "Score", "hu": "Pontszám"},
    "debug_viz_similarity":     {"en": "Similarity", "hu": "Hasonlóság"},
    "debug_viz_probability":    {"en": "Probability", "hu": "Valószínűség"},
    "debug_viz_margin":         {"en": "Margin", "hu": "Különbség"},
    "debug_viz_mode":           {"en": "Mode", "hu": "Mód"},
    "debug_viz_emb_norm":       {"en": "Emb. norm", "hu": "Emb. norma"},
    "debug_viz_reason_assigned":    {"en": "ASSIGNED", "hu": "HOZZÁRENDELVE"},
    "debug_viz_reason_outlier":     {"en": "REJECTED — outlier", "hu": "ELUTASÍTVA — idegen"},
    "debug_viz_reason_threshold":   {"en": "REJECTED — low probability",
                                     "hu": "ELUTASÍTVA — alacsony valószínűség"},
    "debug_viz_reason_margin":      {"en": "REJECTED — ambiguous (margin)",
                                     "hu": "ELUTASÍTVA — kétértelmű (margin)"},
    "debug_viz_reason_prototype":   {"en": "REJECTED — too far from examples",
                                     "hu": "ELUTASÍTVA — túl messze a példáktól"},
    "debug_viz_reason_untrained":   {"en": "SKIPPED — model not trained",
                                     "hu": "KIHAGYVA — modell nincs betanítva"},
    "debug_viz_reason_no_embedding": {"en": "SKIPPED — no embedding",
                                      "hu": "KIHAGYVA — nincs embedding"},
    "debug_viz_reason_unknown_class": {"en": "REJECTED — looks like a stranger",
                                       "hu": "ELUTASÍTVA — ismeretlennek tűnik"},
    "debug_viz_reason_knn":         {"en": "REJECTED — nearest faces disagree",
                                     "hu": "ELUTASÍTVA — a legközelebbi arcok mást mutatnak"},
    "debug_viz_reason_correction":  {"en": "REJECTED — you said it is not this person",
                                     "hu": "ELUTASÍTVA — jelezted, hogy nem ez a személy"},
    "debug_viz_reason_date":        {"en": "REJECTED — photo outside the person's lifetime",
                                     "hu": "ELUTASÍTVA — a fotó a személy életén kívül esik"},
    "debug_viz_reason_repeat":      {"en": "REJECTED — person already on this photo",
                                     "hu": "ELUTASÍTVA — a személy már szerepel a képen"},
    "debug_viz_flow_title":         {"en": "Decision path", "hu": "Döntési útvonal"},
    "debug_viz_nav_live_tooltip":   {"en": "Jump to latest face (live follow mode)",
                                     "hu": "Ugrás a legutóbbi archoz (élő követési mód)"},

    # ── Neural network graph window ───────────────────────────────────────
    "debug_nn_title":         {"en": "Neural Network Graph", "hu": "Neurális háló gráf"},
    "debug_nn_open_btn":      {"en": "Neural Graph", "hu": "Neurális gráf"},
    "debug_nn_open_tooltip":  {"en": "Open full neural-network activation graph",
                               "hu": "Megnyitja a teljes neurális háló aktiváció gráfot"},
    "debug_nn_no_data":       {"en": "No activation data available.\nRun an AI scan with debug mode enabled.",
                               "hu": "Nincs aktivációs adat.\nFuttass AI szkennelést debug módban."},
    "debug_nn_waiting":       {"en": "Waiting for activation data — run an AI scan with debug mode enabled.",
                               "hu": "Aktivációs adatra vár — futtass AI szkennelést debug módban."},
    "debug_nn_group":         {"en": "Neural Network Graph", "hu": "Neurális háló gráf"},
    "debug_nn_group_desc":    {"en": "Open a live graph of the neural network's layer activations. Updates during an AI scan when debug mode is enabled.",
                               "hu": "Megnyitja a neurális háló rétegaktivációinak élő gráfját. AI szkennelés közben frissül, ha a debug mód be van kapcsolva."},

    "export_scope":             {"en": "Scope", "hu": "Hatókör"},
    "export_all_persons": {"en": "All persons", "hu": "Összes személy"},
    "export_selected_person": {"en": "Selected person only: {name}",
                               "hu": "Csak a kiválasztott: {name}"},
    "csv_export":        {"en": "CSV export", "hu": "CSV export"},
    "export_csv_desc":   {"en": "Metadata table: person, face, bounding box, confidence.",
                          "hu": "Metaadat táblázat: személy, arc, bounding box, konfidencia."},
    "export_save_csv":   {"en": "Save CSV …", "hu": "CSV mentése…"},
    "json_export":       {"en": "JSON export", "hu": "JSON export"},
    "export_json_desc":  {"en": "Structured JSON: faces grouped per person.",
                          "hu": "Strukturált JSON: személyenként csoportosított arcok."},
    "export_save_json":  {"en": "Save JSON …", "hu": "JSON mentése…"},
    "export_images_desc": {"en": "Copy face crop thumbnails to a folder.",
                           "hu": "Arc kivágások másolása egy mappába."},
    "export_choose_folder": {"en": "Choose Folder …", "hu": "Mappa kiválasztása…"},
    "export_need_person_tip": {"en": "Select a person in the main window",
                               "hu": "Válassz ki egy személyt a főablakban"},
    "export_astro_group": {"en": "Fast Static Site (Astro)", "hu": "Gyors statikus weboldal (Astro)"},
    "export_astro_desc":  {"en": "Paginated, lazy-loading gallery built with Astro — stays fast with thousands of persons/photos. Generates thumbnails and per-person/per-photo pages. Requires Node.js to build.",
                          "hu": "Lapozott, lazy-loadolt galéria Astro-val — több ezer személy/kép mellett is gyors. Thumbnailöket és személyenkénti/képenkénti oldalakat generál. A buildhez Node.js szükséges."},
    "export_generate_astro": {"en": "Generate Astro Site …", "hu": "Astro weboldal generálása…"},
    "export_local_server_group": {"en": "Local Web Server", "hu": "Helyi webszerver"},
    "export_local_server_desc":  {"en": "Packages the gallery as a password-protected local server (email + OTP via Gmail). Run anywhere with Python + Node.js — no Docker needed. Includes a cross-platform start.py launcher.",
                                  "hu": "A galériát jelszóval védett helyi szerverként csomagolja (Gmail OTP bejelentkezés). Python + Node.js-sel futtatható bárhol, Docker nélkül. Tartalmaz egy platformfüggetlen start.py indítót."},
    "export_generate_local_server": {"en": "Local Server Export …", "hu": "Helyi szerver export…"},
    "local_server_export_title":    {"en": "Local Server Export", "hu": "Helyi szerver export"},
    "task_local_server_export":     {"en": "Local server export", "hu": "Helyi szerver export"},
    "local_server_export_done_title": {"en": "Local Server Export Ready", "hu": "Helyi szerver export kész"},
    "local_server_export_done_msg":   {"en": "Export complete:\n{path}\n\nRun with:\n  python start.py",
                                       "hu": "Export kész:\n{path}\n\nIndítás:\n  python start.py"},
    "local_server_extra_admins":    {"en": "Extra admins (optional)", "hu": "További adminok (opcionális)"},
    "local_server_extra_admins_ph": {"en": "admin2@gmail.com, admin3@gmail.com", "hu": "admin2@gmail.com, admin3@gmail.com"},
    "local_server_extra_admins_info": {
        "en": "Comma-separated extra admin emails. All admins can manage emails, admins and updates; only the primary (email-sending) admin above cannot be removed. Admins must also be able to log in (add them to the allowlist CSV too).",
        "hu": "Vesszővel elválasztott további admin emailek. Minden admin kezelheti az emaileket, adminokat és a frissítést; csak a fenti elsődleges (email-küldő) admin nem törölhető. Az adminoknak be is kell tudni lépniük (vedd fel őket az engedélyező CSV-be is).",
    },
    # --- Update package (uploadable to a running local server) ---
    "export_update_pkg_group": {"en": "Update Package (for a running server)", "hu": "Frissítő csomag (futó szerverhez)"},
    "export_update_pkg_desc":  {"en": "Builds a single .zip with the freshly generated site. Upload it on the running server's /admin page to refresh the gallery without re-deploying. The server can't build — this package carries the pre-built site.",
                               "hu": "Egyetlen .zip-et készít a frissen generált oldallal. Töltsd fel a futó szerver /admin oldalán, hogy újratelepítés nélkül frissítsd a galériát. A szerver nem tud buildelni — ez a csomag a kész oldalt viszi."},
    "export_update_pkg_include_csv": {"en": "Include a new allowed-emails list (CSV)", "hu": "Új engedélyezett email-lista csatolása (CSV)"},
    "export_generate_update_pkg": {"en": "Generate Update Package …", "hu": "Frissítő csomag generálása…"},
    "task_update_pkg_export": {"en": "Update package export", "hu": "Frissítő csomag export"},
    "export_update_pkg_done_title": {"en": "Update Package Ready", "hu": "Frissítő csomag kész"},
    "export_update_pkg_done_msg":   {"en": "Package created:\n{path}\n\nUpload it on the server's /admin page → Update.",
                                     "hu": "A csomag elkészült:\n{path}\n\nTöltsd fel a szerver /admin oldalán → Frissítés."},
    "export_tab_general": {"en": "General", "hu": "Általános"},
    "export_tab_web":     {"en": "Website", "hu": "Weboldal"},
    "local_server_https_mode":   {"en": "HTTPS mode", "hu": "HTTPS mód"},
    "local_server_domain":       {"en": "Domain name", "hu": "Domain név"},
    "local_server_https_none":   {"en": "No HTTPS (LAN / localhost only)", "hu": "Nincs HTTPS (csak LAN / localhost)"},
    "local_server_https_apache": {"en": "Apache / XAMPP + Let's Encrypt (recommended alongside XAMPP)", "hu": "Apache / XAMPP + Let's Encrypt (ajánlott XAMPP mellé)"},
    "local_server_https_caddy":  {"en": "Caddy + Let's Encrypt (Caddy manages ports 80/443)", "hu": "Caddy + Let's Encrypt (Caddy kezeli a 80/443-as portokat)"},
    "local_server_https_cloudflare": {"en": "Cloudflare Tunnel (no open ports — works behind CGNAT / mobile)", "hu": "Cloudflare Tunnel (nincs nyitott port — CGNAT / mobilnet mögött is megy)"},
    "local_server_err_domain":   {"en": "Domain name is required when HTTPS is enabled.", "hu": "HTTPS módhoz domain megadása kötelező."},
    "local_server_smtp_insecure": {"en": "Accept self-signed SMTP certificate (antivirus / proxy)", "hu": "Önaláírt SMTP tanúsítvány elfogadása (vírusirtó / proxy)"},
    "local_server_smtp_insecure_info": {
        "en": "Enable only if email sending fails with 'self signed certificate in certificate chain' — typically when antivirus/firewall inspects TLS. Relaxes certificate checks for the Gmail connection.",
        "hu": "Csak akkor kapcsold be, ha a levélküldés 'self signed certificate in certificate chain' hibával bukik — jellemzően vírusirtó/tűzfal TLS-vizsgálatakor. Lazítja a Gmail-kapcsolat tanúsítvány-ellenőrzését.",
    },
    "local_server_https_info_apache": {
        "en": "Generates apache-vhost.conf + setup-https.sh (Linux) + setup-https.ps1 (Windows). Runs alongside XAMPP: Apache proxies the gallery subdomain to Node.js, certbot/win-acme handles Let's Encrypt. No port conflict.",
        "hu": "apache-vhost.conf + setup-https.sh (Linux) + setup-https.ps1 (Windows) generálódik. XAMPP mellé illeszkedik: Apache proxyzza a galéria aldomént a Node.js-re, a certbot/win-acme kezeli a Let's Encryptet. Nincs port-ütközés.",
    },
    "local_server_https_info_caddy": {
        "en": "Generates Caddyfile + setup-https.sh (Linux) + setup-https.ps1 (Windows). Caddy takes over ports 80/443 and handles Let's Encrypt automatically. If XAMPP runs on 80/443, move it to port 8080 first.",
        "hu": "Caddyfile + setup-https.sh (Linux) + setup-https.ps1 (Windows) generálódik. A Caddy átveszi a 80/443-as portokat és automatikusan kezeli a Let's Encryptet. Ha a XAMPP a 80/443-on fut, előbb tedd át 8080-ra.",
    },
    "local_server_https_info_cloudflare": {
        "en": "Generates setup-tunnel.sh (Linux/macOS) + setup-tunnel.ps1 (Windows). Public HTTPS with NO open ports and NO Let's Encrypt — works behind CGNAT / mobile internet / home router. Requires the domain to be on Cloudflare (free account). start.py then launches the tunnel alongside Node, no sudo needed.",
        "hu": "setup-tunnel.sh (Linux/macOS) + setup-tunnel.ps1 (Windows) generálódik. Nyilvános HTTPS nyitott portok és Let's Encrypt NÉLKÜL — működik CGNAT / mobilnet / otthoni router mögött is. A domén legyen a Cloudflare-en (ingyenes fiók elég). A start.py a Node mellett indítja a tunnelt is, sudo nélkül.",
    },
    # --- Server profile save / load dialogs ---
    "sps_title":                {"en": "Save Server Profile", "hu": "Szerverprofil mentése"},
    "sps_name_label":           {"en": "Profile name", "hu": "Profil neve"},
    "sps_name_placeholder":     {"en": "e.g. Home server", "hu": "pl. Otthoni szerver"},
    "sps_pin_label":            {"en": "PIN (optional)", "hu": "PIN (opcionális)"},
    "sps_pin_placeholder":      {"en": "Leave empty for no PIN", "hu": "Hagyja üresen, ha nem kell PIN"},
    "sps_pin_confirm_label":    {"en": "Confirm PIN", "hu": "PIN megerősítése"},
    "sps_pin_confirm_placeholder": {"en": "Repeat PIN", "hu": "PIN ismétlése"},
    "sps_pin_info":             {"en": "PIN-protected profiles require authentication before loading. Without a PIN the profile is protected only by the OS keychain.", "hu": "PIN-nel védett profilhoz betöltés előtt azonosítás szükséges. PIN nélkül csak az OS kulcstárca védi az adatokat."},
    "sps_macos_secure_info":    {"en": "🔐 This profile will be stored in the macOS Keychain with Touch ID / Face ID protection. Every time you load it, macOS will ask for biometric authentication or your login password.", "hu": "🔐 Ez a profil a macOS kulcstárcában kerül tárolásra Touch ID / Face ID védelemmel. Minden betöltéskor a macOS biometrikus azonosítást vagy belépési jelszót kér."},
    "spl_macos_touchid_info":   {"en": "Touch ID / Face ID will be required when loading. This is enforced by macOS — no PIN needed.", "hu": "A betöltéshez Touch ID / Face ID szükséges. Ezt a macOS kényszeríti ki — nem kell PIN kód."},
    "sps_err_no_name":          {"en": "Profile name is required.", "hu": "A profil nevét meg kell adni."},
    "sps_err_pin_mismatch":     {"en": "The two PINs do not match.", "hu": "A két PIN kód nem egyezik."},
    "sps_err_pin_short":        {"en": "PIN must be at least 4 characters.", "hu": "A PIN legalább 4 karakter kell legyen."},
    "sps_overwrite_confirm":    {"en": "A profile named '{name}' already exists. Overwrite it?", "hu": "Már létezik '{name}' nevű profil. Felülírja?"},
    "spl_title":                {"en": "Load Server Profile", "hu": "Szerverprofil betöltése"},
    "spl_select_label":         {"en": "Select profile", "hu": "Profil kiválasztása"},
    "spl_no_profiles":          {"en": "No saved profiles found.", "hu": "Nincsenek mentett profilok."},
    "spl_biometric_btn":        {"en": "Fingerprint / Face ID", "hu": "Ujjlenyomat / Face ID"},
    "spl_bio_trying":           {"en": "Waiting for biometric…", "hu": "Biometrikus azonosítás…"},
    "spl_bio_ok":               {"en": "Authenticated.", "hu": "Azonosítás sikeres."},
    "spl_bio_failed":           {"en": "Biometric authentication failed.", "hu": "Biometrikus azonosítás nem sikerült."},
    "spl_bio_reason":           {"en": "Face Recognizer — load server profile", "hu": "Arcfelismerő — szerverprofil betöltése"},
    "spl_or_pin":               {"en": "— or enter PIN —", "hu": "— vagy adja meg a PIN-t —"},
    "spl_pin_placeholder":      {"en": "PIN code", "hu": "PIN kód"},
    "spl_load_btn":             {"en": "Load", "hu": "Betöltés"},
    "spl_load_pin_btn":         {"en": "Verify PIN & Load", "hu": "PIN ellenőrzés és betöltés"},
    "spl_delete_btn":           {"en": "Delete", "hu": "Törlés"},
    "spl_cancel_btn":           {"en": "Cancel", "hu": "Mégse"},
    "spl_err_enter_pin":        {"en": "Please enter the PIN code.", "hu": "Adja meg a PIN kódot."},
    "spl_err_wrong_pin":        {"en": "Wrong PIN code.", "hu": "Helytelen PIN kód."},
    "spl_err_load_failed":      {"en": "Could not decrypt profile. The keychain entry may be missing.", "hu": "A profil visszafejtése nem sikerült. Lehet, hogy a kulcstárca bejegyzés hiányzik."},
    "spl_delete_confirm":       {"en": "Delete profile '{name}'?", "hu": "Törli a(z) '{name}' profilt?"},
    "server_profile_save_btn":  {"en": "Save Profile…", "hu": "Profil mentése…"},
    "server_profile_load_btn":  {"en": "Load Profile…", "hu": "Profil betöltése…"},
    "server_profile_saved_msg": {"en": "Profile '{name}' saved.", "hu": "A(z) '{name}' profil mentve."},

    "export_docker_group":   {"en": "Docker / Kubernetes Web Server", "hu": "Docker / Kubernetes webszerver"},
    "export_docker_desc":    {"en": "Packages the gallery as a Docker image with a login page (email + OTP via Gmail). Includes Kubernetes manifests and a single deploy.sh for Linux servers.",
                              "hu": "A galériát Docker image-ként csomagolja be, Gmail OTP-s belépési oldallal. Kubernetes manifesteket és egy deploy.sh szkriptet tartalmaz Linux szerverekhez."},
    "export_generate_docker": {"en": "Docker / K8s Export …", "hu": "Docker / K8s export…"},
    "docker_export_title":       {"en": "Docker / Kubernetes Export", "hu": "Docker / Kubernetes export"},
    "docker_export_admin_email": {"en": "Admin Gmail address", "hu": "Admin Gmail cím"},
    "docker_export_app_password": {"en": "Gmail App Password", "hu": "Gmail App-jelszó"},
    "smtp_test_btn": {"en": "Test", "hu": "Teszt"},
    "smtp_test_title": {"en": "Email test", "hu": "Email teszt"},
    "smtp_test_need_input": {
        "en": "Enter the admin Gmail address and the App Password first.",
        "hu": "Előbb add meg az admin Gmail címet és az App-jelszót.",
    },
    "smtp_test_ok": {
        "en": "Success — a test email was sent to {to}. Check your inbox.",
        "hu": "Sikeres — teszt email elküldve ide: {to}. Nézd meg a postafiókod.",
    },
    "smtp_check_failed": {
        "en": "The Gmail login could not be verified:\n\n{detail}\n\nGenerate anyway?",
        "hu": "A Gmail belépést nem sikerült ellenőrizni:\n\n{detail}\n\nMégis legenerálod?",
    },
    "docker_export_allowed_csv": {"en": "Allowed emails (CSV)", "hu": "Engedélyezett emailek (CSV)"},
    "docker_export_csv_placeholder": {"en": "Select CSV file…", "hu": "CSV fájl kiválasztása…"},
    "docker_export_csv_dialog_title": {"en": "Select CSV with allowed emails", "hu": "CSV fájl az engedélyezett emailekkel"},
    "docker_export_domain":          {"en": "Domain (for K8s Ingress)", "hu": "Domain (K8s Ingress-hez)"},
    "docker_export_cluster_issuer":  {"en": "cert-manager ClusterIssuer", "hu": "cert-manager ClusterIssuer"},
    "docker_export_port":            {"en": "Container port", "hu": "Konténer portja"},
    "docker_export_info":  {"en": "⚠ The generated config.json contains your Gmail App Password — do not push the image to a public registry.",
                            "hu": "⚠ A generált config.json tartalmazza a Gmail App-jelszót — ne töltsd fel a képet nyilvános registry-be."},
    "docker_export_err_email":    {"en": "Admin email must be a Gmail address.", "hu": "Az admin email Gmail cím kell legyen."},
    "docker_export_err_password": {"en": "Gmail App Password is required.", "hu": "A Gmail App-jelszó megadása kötelező."},
    "docker_export_err_csv":      {"en": "Please select a valid CSV file.", "hu": "Válassz egy érvényes CSV fájlt."},
    "docker_export_err_domain":   {"en": "Domain name is required.", "hu": "A domain megadása kötelező."},
    "task_docker_export":         {"en": "Docker/K8s export", "hu": "Docker/K8s export"},
    "docker_export_done_title":   {"en": "Docker Export Ready", "hu": "Docker export kész"},
    "docker_export_done_msg":     {"en": "Export complete:\n{path}\n\nUpload it to your server and run:\n  bash deploy.sh",
                                   "hu": "Export kész:\n{path}\n\nTöltsd fel a szerverre és futtasd:\n  bash deploy.sh"},
    "astro_building": {"en": "Building the Astro site (this may take a minute) …",
                       "hu": "Astro weboldal építése (ez eltarthat egy percig)…"},
    "astro_done": {"en": "Astro Site Ready", "hu": "Astro weboldal kész"},
    "astro_open": {"en": "Generated:\n{path}\n\nOpen it in the browser?",
                   "hu": "Generálva:\n{path}\n\nMegnyitod a böngészőben?"},
    "astro_folder": {"en": "Astro Site Folder", "hu": "Astro weboldal mappája"},
    "astro_failed": {"en": "Astro Export Failed", "hu": "Astro export sikertelen"},
    "astro_need_node": {"en": "Node.js/npm was not found. Install Node.js (https://nodejs.org) to build the Astro site.\n\nDetails: {err}",
                        "hu": "A Node.js/npm nem található. Telepítsd a Node.js-t (https://nodejs.org) az Astro weboldal építéséhez.\n\nRészletek: {err}"},
    "csv_exported":      {"en": "CSV Exported", "hu": "CSV exportálva"},
    "json_exported":     {"en": "JSON Exported", "hu": "JSON exportálva"},
    "json_saved":        {"en": "Saved to:\n{path}", "hu": "Mentve:\n{path}"},
    "images_exported":   {"en": "Images Exported", "hu": "Képek exportálva"},
    "files_copied":      {"en": "{n} file(s) copied:\n{folder}", "hu": "{n} fájl másolva:\n{folder}"},

    # ── Settings dialog ───────────────────────────────────────────────────────
    "settings_title":    {"en": "Settings", "hu": "Beállítások"},
    "lang_label":        {"en": "Language:", "hu": "Nyelv:"},
    "db_group":          {"en": "Database", "hu": "Adatbázis"},
    "current_db":        {"en": "Current:", "hu": "Jelenlegi:"},
    "new_db":            {"en": "New Database …", "hu": "Új adatbázis …"},
    "open_db":           {"en": "Open Database …", "hu": "Adatbázis megnyitása …"},
    "db_switched":       {"en": "Database switched. Please restart the scan.",
                          "hu": "Az adatbázis megváltozott. Indítsa el újra a beolvasást."},
    "db_new_title":      {"en": "Create New Database", "hu": "Új adatbázis létrehozása"},
    "db_open_title":     {"en": "Open Database", "hu": "Adatbázis megnyitása"},
    "updates_group":     {"en": "Updates", "hu": "Frissítések"},
    "current_version":   {"en": "Current version: <b>v{version}</b>",
                          "hu": "Jelenlegi verzió: <b>v{version}</b>"},
    "notify_updates":    {"en": "Notify when a new version is available",
                          "hu": "Értesítés ha új verzió érhető el"},
    "check_updates_btn": {"en": "Check for updates", "hu": "Frissítés keresése"},
    "could_not_reach_github": {"en": "Could not reach GitHub", "hu": "Nem sikerült a kapcsolódás"},
    "up_to_date_version": {"en": "Up to date — v{version}", "hu": "Naprakész — v{version}"},
    "new_version_status": {"en": "New version: <b>v{new}</b> (current: v{current})",
                           "hu": "Új verzió: <b>v{new}</b> (jelenlegi: v{current})"},
    "fix":               {"en": "Fix", "hu": "Javítás"},
    "ai_edge_missing":   {"en": "ai-edge-litert missing", "hu": "ai-edge-litert hiányzik"},
    "libedgetpu_missing_short": {"en": "libedgetpu missing", "hu": "libedgetpu hiányzik"},

    # ── TPU status dialog ─────────────────────────────────────────────────────
    "tpu_title":         {"en": "Google Coral TPU Status", "hu": "Google Coral TPU Állapot"},
    "tpu_devices":       {"en": "Connected devices:", "hu": "Csatlakoztatott eszközök:"},
    "tpu_none":          {"en": "No Edge TPU device found.", "hu": "Nem található Edge TPU eszköz."},
    "tpu_pycoral_ok":    {"en": "pycoral: installed ({ver})", "hu": "pycoral: telepítve ({ver})"},
    "tpu_pycoral_miss":  {"en": "pycoral: NOT installed", "hu": "pycoral: NINCS telepítve"},
    "tpu_ai_edge_missing": {"en": "ai-edge-litert: not installed",
                            "hu": "ai-edge-litert: nincs telepítve"},
    "tpu_delegate_loaded": {"en": "EdgeTPU delegate loaded successfully",
                            "hu": "EdgeTPU delegate sikeresen betöltve"},
    "tpu_libedge_ok":    {"en": "libedgetpu: found", "hu": "libedgetpu: megtalálva"},
    "tpu_libedge_miss":  {"en": "libedgetpu: NOT found", "hu": "libedgetpu: NEM található"},
    "tpu_error":         {"en": "Error: {msg}", "hu": "Hiba: {msg}"},
    "tpu_ok_label":      {"en": "TPU ready ✓", "hu": "TPU kész ✓"},
    "tpu_warn_label":    {"en": "TPU not available", "hu": "TPU nem elérhető"},
    "tpu_inference_ok":  {"en": "✓ Test inference succeeded — TPU is actively accelerating detection",
                          "hu": "✓ Teszt következtetés sikeres — a TPU aktívan gyorsítja a detektálást"},
    "tpu_inference_fail": {"en": "✗ Library loads but device is NOT responding. Detection will use CPU.",
                           "hu": "✗ A könyvtár betöltődik, de az eszköz NEM válaszol. A detektálás CPU-n fut."},
    "tpu_phantom_tip":    {"en": "Tip: unplug the Coral USB device, wait 5 seconds, plug it back in, "
                                  "then click 'Re-check' below. On macOS also check:\n"
                                  "  System Settings → Privacy & Security → USB / Accessory Security",
                           "hu": "Tipp: húzd ki a Coral USB eszközt, várj 5 másodpercet, dugd vissza, "
                                  "majd kattints az 'Újraellenőrzés' gombra. macOS-en nézd meg:\n"
                                  "  Rendszerbeállítások → Adatvédelem és biztonság → USB / Tartozék biztonság"},
    "tpu_recheck":        {"en": "🔄 Re-check / Újraellenőrzés",
                           "hu": "🔄 Újraellenőrzés / Re-check"},
    "tpu_checking":       {"en": "Checking...", "hu": "Ellenőrzés..."},
    "tpu_manual_commands": {"en": "Manual install commands (run in terminal)",
                            "hu": "Kézi telepítési parancsok (futtasd terminálban)"},
    "tpu_copy_commands":  {"en": "Copy commands", "hu": "Parancsok másolása"},
    "tpu_auto_fix":       {"en": "Auto-fix", "hu": "Automatikus javítás"},
    "tpu_run_auto_fix":   {"en": "Run auto-fix", "hu": "Automatikus javítás indítása"},
    "tpu_fix_done":       {"en": "\n✓ Done! Restart the app.",
                           "hu": "\n✓ Kész! Indítsa újra az alkalmazást."},
    "tpu_fix_failed":     {"en": "\n✗ Some commands failed — copy the commands above and run in terminal as admin.",
                           "hu": "\n✗ Néhány parancs meghiúsult — másolja ki a parancsokat és futtassa terminálban rendszergazdaként."},

    # ── Image viewer / manual face marking ────────────────────────────────
    "view_all_images":   {"en": "All Images", "hu": "Összes kép"},
    "view_no_face":      {"en": "Images Without Faces", "hu": "Arc nélküli képek"},
    "view_by_person":    {"en": "By Person", "hu": "Személy szerint"},
    "n_images_no_face":  {"en": "{n} image(s) with no detected face",
                          "hu": "{n} kép arc nélkül"},
    "mark_face":         {"en": "Mark Faces Manually", "hu": "Arcok kézi jelölése"},
    "mark_face_hint":    {"en": "Drag on the image to mark a face region. You can add multiple faces.",
                          "hu": "Húzd az egeret a képen az arc jelöléséhez. Több arcot is hozzáadhatsz."},
    "mark_face_saved":   {"en": "Face saved. Assign it to a person from the sidebar.",
                          "hu": "Arc mentve. Rendeld személyhez az oldalsávból."},
    "mark_face_add_btn": {"en": "Add Face", "hu": "Arc hozzáadása"},
    "mark_face_done_btn": {"en": "Done", "hu": "Kész"},
    "mark_face_count":   {"en": "{n} face(s) marked", "hu": "{n} arc jelölve"},
    "mark_face_none_yet": {"en": "Draw a rectangle to mark a face",
                           "hu": "Rajzolj téglalapot egy arc jelöléséhez"},
    "previous":          {"en": "Previous", "hu": "Előző"},
    "next":              {"en": "Next", "hu": "Következő"},
    "manual_marking":    {"en": "Manual Marking", "hu": "Kézi jelölés"},
    "manual_marking_tip": {"en": "Click, then drag on the image to manually mark a face",
                           "hu": "Kattints, majd húzd az egeret a képen egy arc kézi megjelöléséhez"},
    "fullscreen":        {"en": "Full Screen", "hu": "Teljes képernyő"},
    "fullscreen_tip":    {"en": "Full-screen view", "hu": "Teljes képernyős nézet"},
    "exit_fullscreen":   {"en": "Exit Full Screen", "hu": "Kilépés a teljes képernyőből"},
    "exit_fullscreen_tip": {"en": "Exit full-screen view", "hu": "Kilépés a teljes képernyős nézetből"},
    "draw_face_hint_auto": {"en": "Drag around the face to mark it — saved automatically",
                            "hu": "Húzd az egeret az arc körül a jelöléshez — mentés automatikus"},
    "draw_face_hint":    {"en": "Drag around the face to select it",
                          "hu": "Húzd az egeret az arc körül a kijelöléshez"},
    "redraw_face_hint":  {"en": "Redraw the selection around the face",
                          "hu": "Rajzold újra az arc körüli kijelölést"},
    "redraw_face_rect_hint": {"en": "Redraw the rectangle around the face — it replaces the old one",
                              "hu": "Rajzold újra az arc körüli téglalapot — a régi helyére kerül"},
    "select_folder_scan": {"en": "Select a folder and run a scan",
                           "hu": "Válassz mappát és futtass beolvasást"},
    "folder":            {"en": "Folder:", "hu": "Mappa:"},
    "image_date_period": {"en": "Image date / period:", "hu": "Kép dátuma / időszaka:"},
    "date_period_placeholder": {"en": "e.g. 1954 or 1954.03.12 or the 1930s",
                                "hu": "pl. 1954 vagy 1954.03.12 vagy 1930-as évek"},
    "date_period_tip":   {"en": "The date or period when the image was taken (free text)",
                          "hu": "A kép készítésének dátuma vagy időszaka (szabad szöveg)"},
    "selected_face":     {"en": "Selected face:", "hu": "Kiválasztott arc:"},
    "click_face_on_image": {"en": "Click a face on the image", "hu": "Kattints egy arcra a képen"},
    "identified_person": {"en": "Identified person:", "hu": "Beazonosított személy:"},
    "face_uncategorized": {"en": "This face is uncategorized", "hu": "Ez az arc nincs kategorizálva"},
    "no_face_on_image":  {"en": "No recognized face on this image", "hu": "Nincs felismert arc ezen a képen"},
    "new_name":          {"en": "New name …", "hu": "Új név…"},
    "assign_existing":   {"en": "Assign to existing person:", "hu": "Hozzárendelés meglévő személyhez:"},
    "assign":            {"en": "Assign", "hu": "Hozzárendelés"},
    "create_person":     {"en": "Create new person:", "hu": "Új személy létrehozása:"},
    "person_name":       {"en": "Person name …", "hu": "Személy neve…"},
    "create_and_assign": {"en": "Create and Assign", "hu": "Létrehozás és hozzárendelés"},
    "no_images_in_db":   {"en": "No images in the database", "hu": "Nincs kép az adatbázisban"},
    "cannot_load":       {"en": "Cannot load:\n{path}", "hu": "Nem tölthető be:\n{path}"},
    "rename_person_tip": {"en": "Rename person", "hu": "Személy átnevezése"},
    "edit_person_info_tip": {"en": "Edit person info", "hu": "Személyadatok szerkesztése"},

}
