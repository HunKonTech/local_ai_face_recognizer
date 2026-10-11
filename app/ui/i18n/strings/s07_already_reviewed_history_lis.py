"""UI strings (EN + HU) — part 7, starting at: "Already reviewed" history lists."""

from __future__ import annotations

from typing import Dict

STRINGS: Dict[str, Dict[str, str]] = {
    # ── "Already reviewed" history lists ──────────────────────────────────
    "autoAssign.decided_header":
        {"en": "Already reviewed ({n})",
         "hu": "Már átnézett ({n})"},
    "autoAssign.decided_empty":
        {"en": "No reviewed groupings yet.",
         "hu": "Még nincs átnézett csoportosítás."},
    "autoAssign.status_confirmed":
        {"en": "Confirmed",
         "hu": "Megerősítve"},
    "autoAssign.status_corrected":
        {"en": "Corrected → {name}",
         "hu": "Javítva → {name}"},
    "autoAssign.status_reverted":
        {"en": "Undone",
         "hu": "Visszavonva"},
    "autoAssign.sort_label":
        {"en": "Order:",
         "hu": "Sorrend:"},
    "autoAssign.sort_uncertain":
        {"en": "Least certain first",
         "hu": "Legbizonytalanabb elöl"},
    "autoAssign.sort_certain":
        {"en": "Most certain first",
         "hu": "Legbiztosabb elöl"},
    "autoAssign.sort_tooltip":
        {"en": "Reviewing the least certain groupings first teaches the AI the most.",
         "hu": "A legbizonytalanabb csoportosítások átnézése tanítja a legtöbbet az AI-nak."},
    "autoAssign.near_header":
        {"en": "Near matches — the AI was unsure ({n})",
         "hu": "Majdnem-találatok — az AI bizonytalan volt ({n})"},
    "autoAssign.near_intro":
        {"en": "These faces were NOT assigned: the AI found a likely person but was "
               "not sure enough. One click confirms or rejects — each answer is "
               "exactly the example the AI needs most.",
         "hu": "Ezeket az arcokat az AI NEM rendelte hozzá: talált valószínű személyt, "
               "de nem volt elég biztos. Egy kattintással megerősítheted vagy "
               "elutasíthatod — minden válasz pont az a példa, amire az AI-nak a "
               "legnagyobb szüksége van."},
    "autoAssign.near_empty":
        {"en": "No near matches in the last run.",
         "hu": "Az utolsó futásban nem volt majdnem-találat."},
    "autoAssign.near_confirm":
        {"en": "Yes, it's them",
         "hu": "Igen, ő az"},
    "autoAssign.near_confirm_tooltip":
        {"en": "Assign the face to this person (trusted training example).",
         "hu": "Az arc hozzárendelése ehhez a személyhez (megbízható tanítópélda)."},
    "autoAssign.near_reject":
        {"en": "Not them",
         "hu": "Nem ő"},
    "autoAssign.near_reject_tooltip":
        {"en": "The face stays where it is; the AI will never suggest this person for it again.",
         "hu": "Az arc marad, ahol van; az AI soha többé nem ajánlja hozzá ezt a személyt."},
    "autoAssign.suspicious_header":
        {"en": "Suspicious labels ({n})",
         "hu": "Gyanús címkék ({n})"},
    "autoAssign.suspicious_intro":
        {"en": "These faces are labeled with a person, but look much more like "
               "someone else. They are left out of AI training until you decide.",
         "hu": "Ezek az arcok egy személyhez vannak rendelve, de sokkal inkább "
               "valaki másra hasonlítanak. Amíg nem döntesz, kimaradnak az AI "
               "tanításából."},
    "autoAssign.suspicious_empty":
        {"en": "No suspicious labels.",
         "hu": "Nincs gyanús címke."},
    "autoAssign.suspicious_lookalike":
        {"en": "{name} — looks more like {other}",
         "hu": "{name} — inkább {other} személyre hasonlít"},
    "autoAssign.suspicious_keep":
        {"en": "Label is right",
         "hu": "A címke jó"},
    "autoAssign.suspicious_keep_tooltip":
        {"en": "Keep the label and use the face for training again.",
         "hu": "A címke marad, és az arc újra tanítóadat lesz."},
    "autoAssign.suspicious_remove":
        {"en": "Remove label",
         "hu": "Címke eltávolítása"},
    "autoAssign.suspicious_remove_tooltip":
        {"en": "Send the face back to the unknown faces; the AI will not put it on this person again.",
         "hu": "Az arc visszakerül az ismeretlenek közé; az AI nem teszi vissza erre a személyre."},
    "autoAssign.metrics":
        {"en": "AI calibration: target precision {target}, measured from your reviews {measured}",
         "hu": "AI kalibráció: cél-pontosság {target}, az átnézéseid alapján mért {measured}"},
    "scanModes.openButton":
        {"en": "Scan & Maintenance …",
         "hu": "Beolvasás és karbantartás …"},
    "scanModes.close":
        {"en": "Close",
         "hu": "Bezárás"},
    "scanModes.tab.ai":
        {"en": "AI recognition",
         "hu": "AI felismerés"},
    "scanModes.tab.maintenance":
        {"en": "Maintenance (developer)",
         "hu": "Karbantartás (fejlesztői)"},
    "scanModes.maintenance.note":
        {"en": "These cleanup and repair tools now run automatically as part of "
               "the redesigned detection pipeline (during scan / rescan). They are "
               "kept here so they can still be launched manually for debugging or "
               "one-off fixes — a developer convenience.",
         "hu": "Ezek a tisztító és helyreállító eszközök az újratervezett "
               "arcdetektálási folyamat részeként már automatikusan lefutnak "
               "(beolvasáskor / újraolvasáskor). Itt azért maradnak meg, hogy "
               "hibakereséshez vagy egyszeri javításhoz manuálisan is "
               "elindíthatók legyenek — ez fejlesztői kényelmi funkció."},
    "scanModes.verifyAll.label":
        {"en": "Verify every face with multiple technologies (no confidence threshold)",
         "hu": "Minden arc ellenőrzése több technológiával (nincs konfidencia-küszöb)"},
    "scanModes.verifyAll.tip":
        {"en": "When on, the multi-technology face check (YuNet + Caffe + Haar + eye "
               "cascade + InsightFace) runs on EVERY detected face, not only the "
               "uncertain ones — so high-confidence false positives (ears, the back "
               "of a head, objects) are caught and removed too. Slower. Applies to "
               "detection, Full Rescan and the stored-face re-check. Run a Full "
               "Rescan afterwards to re-process already-scanned images.",
         "hu": "Bekapcsolva a több-technológiás arc-ellenőrzés (YuNet + Caffe + Haar + "
               "szem-kaszkád + InsightFace) MINDEN felismert arcon lefut, nem csak a "
               "bizonytalanokon — így a magas konfidenciájú fals pozitívok (fül, "
               "tarkó, tárgyak) is kiszűrődnek. Lassabb. A detektálásra, a Teljes "
               "újrabeolvasásra és a tárolt arcok újra-ellenőrzésére is vonatkozik. "
               "A már beolvasott képekhez futtass utána Teljes újrabeolvasást."},

    # incremental (scan & index)
    "scanModes.incremental.title":
        {"en": "Scan & Index",
         "hu": "Beolvasás és indexelés"},
    "scanModes.incremental.description":
        {"en": "Scans the selected folder for new images that are not yet in the database. "
               "Runs face detection, embedding and recognition only on these new files. "
               "Existing faces and person assignments are left completely untouched. "
               "Fast and safe — ideal for daily incremental updates after adding new photos.",
         "hu": "Csak az adatbázisban még nem szereplő új képeket dolgozza fel. "
               "Arcdetektálást, embeddinget és felismerést csak az új fájlokon futtat. "
               "A meglévő arcok és személyesítések érintetlenek maradnak. "
               "Gyors és biztonságos — mindennapi használatra, ha új fotókat adtál hozzá."},
    "scanModes.incremental.startButton":
        {"en": "Start Scan",
         "hu": "Beolvasás indítása"},

    # reset automatically created Unknown identities
    "scanModes.resetUnknowns.title":
        {"en": "Rebuild Unknown Identities",
         "hu": "Unknown személyek újraépítése"},
    "scanModes.resetUnknowns.description":
        {"en": "Deletes every automatically created 'Unknown N' person, makes their faces "
               "unassigned again and re-clusters them into fresh Unknown groups "
               "(fast -- no image scanning or model training). Face boxes and embeddings "
               "are preserved unless you tick the face-deletion step below. Named and "
               "protected people are never touched.",
         "hu": "Törli az összes automatikusan létrehozott 'Unknown N' személyt, az arcaikat "
               "újra hozzárendeletlen állapotba teszi, majd friss Unknown csoportokba "
               "klaszterezi őket (gyors - nem fut újra a detektálás vagy a tanítás). Az "
               "arckeretek és embeddingek megmaradnak, hacsak be nem jelöli az alábbi "
               "arctörlési lépést. Az elnevezett és védett személyeket soha nem érinti."} ,
    "scanModes.resetUnknowns.startButton":
        {"en": "Rebuild Unknown Identities",
         "hu": "Unknown személyek újraépítése"},
    "scanModes.resetUnknowns.warning":
        {"en": "Auto-created Unknown groups will be deleted and rebuilt",
         "hu": "Az automatikus Unknown csoportok törlődnek és újraépülnek"},
    "reset_unknowns_status":
        {"en": "Rebuilding Unknown identities: {persons} person(s) deleted, "
               "{faces} face(s) reset, {deleted} face(s) deleted",
         "hu": "Unknown személyek újraépítése: {persons} személy törölve, "
               "{faces} arc alaphelyzetbe állítva, {deleted} arc törölve"},

    # Reset Unknown Persons options (inline in the Scan & Maintenance dialog)
    "resetUnknownOptions.deletePersons":
        {"en": "Delete auto-created 'Unknown N' persons",
         "hu": "Automatikusan létrehozott 'Unknown N' személyek törlése"},
    "resetUnknownOptions.deletePersonsTooltip":
        {"en": "Remove the automatically created Unknown person records from the database",
         "hu": "Az automatikusan létrehozott Unknown személy rekordok eltávolítása az adatbázisból"},
    "resetUnknownOptions.deleteFaceAssignments":
        {"en": "Unassign faces from Unknown persons",
         "hu": "Arcok hozzárendelésének eltávolítása Unknown személyektől"},
    "resetUnknownOptions.deleteFaceAssignmentsTooltip":
        {"en": "Remove person assignments, allowing faces to be re-identified",
         "hu": "A személyhozzárendelések eltávolítása, lehetővé téve az arcok újra azonosítását"},
    "resetUnknownOptions.deleteFaceData":
        {"en": "Delete the Unknown persons' faces entirely",
         "hu": "Az Unknown személyek arcainak teljes törlése"},
    "resetUnknownOptions.deleteFaceDataTooltip":
        {"en": "⚠ Advanced: permanently deletes the face records themselves — box, embedding "
               "and crop file. The faces disappear from the images and only a full "
               "re-detection scan can bring them back. Named and protected people are "
               "never touched.",
         "hu": "⚠ Haladó: véglegesen törli magukat az arc rekordokat — keret, embedding és "
               "kivágott kép. Az arcok eltűnnek a képekről, és csak egy teljes újra-detektálás "
               "hozza vissza őket. Az elnevezett és védett személyeket nem érinti."},
    "resetUnknownOptions.rebuildClusters":
        {"en": "Rebuild Unknown clusters after reset",
         "hu": "Unknown klaszterek újraépítése a törlés után"},
    "resetUnknownOptions.rebuildClustersTooltip":
        {"en": "After deleting Unknown persons, automatically re-cluster all unassigned faces into new Unknown groups (fast -- no image scanning or model training)",
         "hu": "Az Unknown személyek törlése után automatikusan újra klaszterezi a nem hozzárendelt arcokat új Unknown csoportokba (gyors - nem fut újra a detektálás vagy a tanítás)"},

    # overlapping question-mark cleanup
    "scanModes.overlapCleanup.title":
        {"en": "Find Overlapping / Intersecting ? Boxes",
         "hu": "Átfedő / metsző ? keretek keresése"},
    "scanModes.overlapCleanup.description":
        {"en": "Searches the current database for unassigned question-mark face boxes "
               "that overlap already named faces. It only lists suspicious "
               "candidates first; nothing is deleted until you review the list, keep the "
               "checkboxes you want, and confirm deletion.",
         "hu": "Megkeresi az adatbázisban azokat a kérdőjeles, személyhez nem rendelt "
               "arckereteket, amelyek átfednek vagy metszenek egy másik arckeretet. "
               "Először csak listázza a gyanús találatokat; semmit nem töröl addig, "
               "amíg át nem nézed a listát, ki nem választod a törlendőket, és meg "
               "nem erősíted a törlést."},
    "scanModes.overlapCleanup.sensitivity":
        {"en": "Sensitivity:", "hu": "Érzékenység:"},
    "scanModes.overlapCleanup.sensitivityTip":
        {"en": "The strict level only lists heavily overlapping boxes. The looser "
               "levels also list boxes that merely intersect, and boxes belonging to "
               "two different identities — more finds, more false candidates to skip "
               "in the review list.",
         "hu": "A szigorú szint csak az erősen átfedő kereteket listázza. A lazább "
               "szintek a csak metsző kereteket is megtalálják, és a két különböző "
               "személyhez tartozó párokat is — több találat, de több téves jelölt is "
               "az átnézendő listán."},
    "overlapSensitivity.strict":
        {"en": "Strict — heavy overlap only (default)",
         "hu": "Szigorú — csak erős átfedés (alapértelmezett)"},
    "overlapSensitivity.medium":
        {"en": "Medium — partial overlap, across identities too",
         "hu": "Közepes — részleges átfedés, más személlyel is"},
    "overlapSensitivity.any":
        {"en": "Loose — every intersecting box pair",
         "hu": "Laza — minden metsző keretpár"},
    "scanModes.overlapCleanup.startButton":
        {"en": "Find Overlapping ? Boxes",
         "hu": "Átfedő / metsző keretek keresése"},
    "scanModes.overlapCleanup.warning":
        {"en": "Review step included — known named faces are never deleted",
         "hu": "Átnézési lépéssel — az ismert, elnevezett arcokat soha nem törli"},

    # ── Embedding-based duplicate cleanup ─────────────────────────────────
    "scanModes.embeddingDuplicates.title":
        {"en": "Find Duplicate Detections (by face similarity)",
         "hu": "Dupla detektálások keresése (arc-hasonlóság alapján)"},
    "scanModes.embeddingDuplicates.description":
        {"en": "Finds the same physical face detected twice on one image — even "
               "when the two boxes are differently sized or landed in two "
               "different 'Unknown' people — by comparing face embeddings, not "
               "just box overlap. The geometric search above misses this case. "
               "Nothing is deleted until you review the list and confirm.",
         "hu": "Megkeresi azokat az eseteket, amikor ugyanazt az arcot egy képen "
               "kétszer detektálta a rendszer — akkor is, ha a két keret eltérő "
               "méretű, vagy két külön „Ismeretlen” személyhez került —, a "
               "puszta keret-átfedés helyett az arc-beágyazások összevetésével. "
               "A fenti geometriai keresés ezt nem fogja meg. Semmit nem töröl, "
               "amíg át nem nézed a listát és meg nem erősíted."},
    "scanModes.embeddingDuplicates.startButton":
        {"en": "Find Duplicate Detections",
         "hu": "Dupla detektálások keresése"},
    "scanModes.embeddingDuplicates.warning":
        {"en": "Review step included — known named faces are never deleted",
         "hu": "Átnézési lépéssel — az ismert, elnevezett arcokat soha nem törli"},

    # ── Identity repair scan ──────────────────────────────────────────────
    "scanModes.identityRepair.title":
        {"en": "Identity Repair Scan",
         "hu": "Identitás-helyreállító vizsgálat"},
    "scanModes.identityRepair.description":
        {"en": "Scans the whole database for 'Unknown N' identities that are "
               "really the same person split apart over many runs (e.g. "
               "Unknown 98 ≈ Unknown 155). It only proposes merges first; "
               "nothing is consolidated until you review the list, keep the "
               "pairs you want, and confirm.",
         "hu": "Átvizsgálja az egész adatbázist olyan „Unknown N” identitások "
               "után, amelyek valójában ugyanaz a személy, csak több futás "
               "során szétestek (pl. Unknown 98 ≈ Unknown 155). Először csak "
               "összevonásokat javasol; semmit nem von össze, amíg át nem nézed "
               "a listát, ki nem választod a párokat, és meg nem erősíted."},
    "scanModes.identityRepair.startButton":
        {"en": "Scan for Fragmented Identities",
         "hu": "Széttöredezett identitások keresése"},
    "scanModes.identityRepair.warning":
        {"en": "Review step included — only auto-named Unknown persons are merged",
         "hu": "Átnézési lépéssel — csak automatikus „Unknown” személyeket von össze"},

    # ── Clean up empty Unknown persons ────────────────────────────────────
    "scanModes.cleanupEmptyUnknowns.title":
        {"en": "Clean Up Empty Unknown Persons",
         "hu": "Üres Unknown személyek tisztítása"},
    "scanModes.cleanupEmptyUnknowns.description":
        {"en": "Removes automatically created 'Unknown N' persons that no longer have "
               "any faces attached. These empty placeholders are left behind when their "
               "last face is reassigned, removed, or excluded, and show up as blank '?' "
               "entries in the person list. Named people and the protected 'Ismeretlen' "
               "person are always kept, and no person that still owns a face is touched. "
               "Face boxes and embeddings are not affected.",
         "hu": "Eltávolítja azokat az automatikusan létrehozott „Unknown N” személyeket, "
               "amelyekhez már egyetlen arc sem tartozik. Ezek az üres helykitöltők akkor "
               "maradnak vissza, amikor az utolsó arcukat átrendelik, eltávolítják vagy "
               "kizárják, és üres „?” bejegyzésként jelennek meg a személylistában. Az "
               "elnevezett személyek és a védett „Ismeretlen” személy mindig megmaradnak, "
               "és egyetlen arccal rendelkező személyt sem érint. Az arckereteket és "
               "embeddingeket nem befolyásolja."},
    "scanModes.cleanupEmptyUnknowns.startButton":
        {"en": "Clean Up Empty Unknown Persons",
         "hu": "Üres Unknown személyek tisztítása"},
    "scanModes.cleanupEmptyUnknowns.warning":
        {"en": "Safe — only face-less auto-named Unknown persons are removed",
         "hu": "Biztonságos — csak az arc nélküli automatikus „Unknown” személyeket törli"},
    "scanModes.ignoredFaces.title":
        {"en": "Ignored Faces",
         "hu": "Kizárt arcok"},
    "scanModes.ignoredFaces.description":
        {"en": "Manage the permanent face ignore list. Faces excluded with 'Ignore "
               "Forever' are remembered by their face signature, so they stay hidden "
               "across re-scans. Here you can review them and revoke entries so a "
               "face can be recognised again.",
         "hu": "A végleges arc-kizárólista kezelése. A „Kizárás végleg” művelettel "
               "kizárt arcokat a rendszer az arc-jellemzőjük alapján jegyzi meg, így "
               "újrafuttatás után is rejtve maradnak. Itt áttekintheted őket, és "
               "visszavonhatod a kizárást, hogy egy arc újra felismerhető legyen."},
    "scanModes.ignoredFaces.startButton":
        {"en": "Manage Ignored Faces",
         "hu": "Kizárt arcok kezelése"},
    "scanModes.reembedFaces.title":
        {"en": "Recompute Face Embeddings (AI)",
         "hu": "Arc-jellemzők újraszámolása (AI)"},
    "scanModes.reembedFaces.description":
        {"en": "Recomputes the embedding of every face with the current settings "
               "(e.g. mirror-image averaging, which makes every face vector cleaner "
               "and records how reliable it is), then retrains the AI model from "
               "scratch and re-recognizes unknown faces. Face boxes and person "
               "assignments are kept.",
         "hu": "Minden arc jellemzővektorát újraszámolja a jelenlegi beállításokkal "
               "(pl. tükörkép-átlagolással, ami minden arcvektort tisztábbá tesz, és "
               "rögzíti, mennyire megbízható), majd nulláról újratanítja az AI "
               "modellt, és újra felismeri az ismeretlen arcokat. Az arckeretek és a "
               "személy-hozzárendelések megmaradnak."},
    "scanModes.reembedFaces.startButton":
        {"en": "Recompute Embeddings",
         "hu": "Jellemzők újraszámolása"},
    "scanModes.reembedFaces.warning":
        {"en": "Slow on large libraries — every face crop is processed again",
         "hu": "Nagy könyvtárnál lassú — minden arcképet újra feldolgoz"},
    "reembed_confirm_title":
        {"en": "Recompute embeddings",
         "hu": "Jellemzők újraszámolása"},
    "reembed_confirm_msg":
        {"en": "Recompute the embedding of every face and retrain the AI model from "
               "scratch?\n\nFace boxes and person assignments are kept. This can "
               "take a long time on a large library.",
         "hu": "Újraszámolja minden arc jellemzővektorát, és nulláról újratanítja "
               "az AI modellt?\n\nAz arckeretek és a személy-hozzárendelések "
               "megmaradnak. Nagy könyvtárnál sokáig tarthat."},
    "cleanup_empty_unknowns_title":
        {"en": "Clean Up Empty Unknown Persons",
         "hu": "Üres Unknown személyek tisztítása"},
    "cleanup_empty_unknowns_msg":
        {"en": "Delete every automatically created 'Unknown' person that has no faces?\n\n"
               "Named people and the protected 'Ismeretlen' person are preserved.",
         "hu": "Törli az összes olyan automatikusan létrehozott „Unknown” személyt, "
               "amelyhez nem tartozik arc?\n\n"
               "Az elnevezett személyek és a védett „Ismeretlen” személy megmaradnak."},
    "cleanup_empty_unknowns_status":
        {"en": "Cleaned up {persons} empty Unknown person(s)",
         "hu": "{persons} üres Unknown személy törölve"},
    "cleanup_empty_unknowns_none":
        {"en": "No empty Unknown persons were found.",
         "hu": "Nem található üres Unknown személy."},

    # ── Re-validate stored faces against landmark geometry ────────────────
    "scanModes.geomCleanup.title":
        {"en": "Remove False Faces (Geometry Re-check)",
         "hu": "Hamis arcok eltávolítása (geometria-ellenőrzés)"},
    "scanModes.geomCleanup.description":
        {"en": "Re-checks the stored facial landmarks of already-detected faces and "
               "removes the geometrically impossible ones — cards, hands, feet and tree "
               "knots that were detected before the landmark-geometry filter existed. No "
               "image is re-scanned; it only re-reads the saved landmarks. Faces you "
               "assigned by hand (and manually drawn boxes) are never deleted — they are "
               "only listed for review. You get a preview with exact counts before "
               "anything is removed.",
         "hu": "Újra ellenőrzi a már detektált arcok elmentett arc-pontjait, és eltávolítja "
               "a geometriailag lehetetleneket — a kártyákat, kezeket, lábakat és "
               "fa-göcsörtöket, amelyeket még a landmark-geometria szűrő előtt rögzített a "
               "rendszer. Nem szkennel újra képet, csak az elmentett pontokat olvassa "
               "vissza. A kézzel hozzárendelt arcokat (és a kézzel rajzolt kereteket) "
               "soha nem törli — azokat csak felülvizsgálatra listázza. A törlés előtt "
               "pontos darabszámot mutató előnézetet kapsz."},
    "scanModes.geomCleanup.startButton":
        {"en": "Re-check & Remove False Faces",
         "hu": "Ellenőrzés és hamis arcok törlése"},
    "scanModes.geomCleanup.warning":
        {"en": "Deletes faces — but only impossible-geometry, non-manual ones, after a "
               "confirmation preview",
         "hu": "Arcokat töröl — de csak a geometriailag lehetetlen, nem kézi arcokat, "
               "megerősítő előnézet után"},
    "geom_cleanup_title":
        {"en": "Remove False Faces",
         "hu": "Hamis arcok eltávolítása"},
    "geom_cleanup_confirm":
        {"en": "Re-checked {scanned} stored face(s) with landmarks.\n\n"
               "{n} face(s) have impossible geometry and can be removed "
               "(unassigned or automatically assigned).\n"
               "{flagged} manually/confirmed face(s) also look impossible — these are "
               "kept for your review and NOT deleted.\n\n"
               "Delete the {n} false face(s) now? This cannot be undone.",
         "hu": "{scanned} elmentett, arc-pontokkal rendelkező arc ellenőrizve.\n\n"
               "{n} arc geometriája lehetetlen és eltávolítható "
               "(hozzárendeletlen vagy automatikusan hozzárendelt).\n"
               "{flagged} kézi/megerősített arc szintén lehetetlennek tűnik — ezeket "
               "felülvizsgálatra megtartjuk, és NEM töröljük.\n\n"
               "Törölhető most a {n} hamis arc? Ez nem vonható vissza."},
    "geom_cleanup_none":
        {"en": "Re-checked {scanned} stored face(s) — no removable false faces found.\n"
               "({flagged} manual/confirmed face(s) look impossible but are kept for "
               "review.)",
         "hu": "{scanned} elmentett arc ellenőrizve — nincs eltávolítható hamis arc.\n"
               "({flagged} kézi/megerősített arc lehetetlennek tűnik, de "
               "felülvizsgálatra megmarad.)"},
    "geom_cleanup_status":
        {"en": "Removed {deleted} false face(s)",
         "hu": "{deleted} hamis arc eltávolítva"},

    # Retroactive crop-level verification
    "scanModes.retroVerify.title":
        {"en": "Remove False Faces (Crop Re-detection)",
         "hu": "Hamis arcok eltávolítása (vágat-szintű újra-detektálás)"},
    "scanModes.retroVerify.description":
        {"en": "Re-examines every already-detected (non-manual) face by loading its "
               "original image and running the crop-level face verifier on the stored "
               "bounding box — the same gate applied to new detections. Faces that "
               "cannot be re-found inside their own crop (objects, ears, walls, fabric) "
               "are removed. Manually drawn boxes and faces you assigned by hand are "
               "never deleted — they are listed for review only. You get a count preview "
               "before anything is removed.",
         "hu": "Minden már detektált (nem kézi) arcot újra megvizsgál: betölti az eredeti "
               "képet, és futtatja a vágat-szintű arc-verifikátort az elmentett "
               "keretezésen — ugyanolyan kaput alkalmaz, mint az új detektáláskor. Azokat "
               "az arcokat, amelyeket nem sikerül újra megtalálni a saját kivágásukon "
               "belül (tárgyak, fülek, falak, szövetek), eltávolítja. A kézzel rajzolt "
               "kereteket és a kézzel hozzárendelt arcokat soha nem törli — csak "
               "felülvizsgálatra listázza. A törlés előtt darabszám-előnézetet kapsz."},
    "scanModes.retroVerify.startButton":
        {"en": "Re-verify & Remove False Faces",
         "hu": "Újra-ellenőrzés és hamis arcok törlése"},
    "scanModes.retroVerify.warning":
        {"en": "Loads every original image — may be slow on large libraries. "
               "Deletes non-confirmed non-face detections after a preview.",
         "hu": "Minden eredeti képet betölt — nagy könyvtárakon lassú lehet. "
               "Megerősítő előnézet után törli a nem megerősített, nem arc detektálásokat."},

}
