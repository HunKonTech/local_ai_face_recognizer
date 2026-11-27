# Face-Local — projekt elemzés

> Állapot: 2026-11-27 · `main` @ `484ba6d` (v1.0.72) · felépítés: [PROJEKT_FELEPITES.md](PROJEKT_FELEPITES.md) · jegyek: [NYITOTT_JEGYEK.md](NYITOTT_JEGYEK.md)

## Számokban

| Mutató | Érték |
|---|---|
| Python forrás (`app/`) | ~105 ezer sor, ~285 fájl |
| Tesztkód (`tests/`) | ~34,5 ezer sor, ~155 tesztfájl |
| Szolgáltatás-modul | 73 (`app/services/`) |
| ORM tábla | 31 |
| 800 sornál hosszabb modul | 29 |
| Commit | 515 (2026-04-16 → 2026-10-13, ~6 hónap) |
| Release | v1.0.72 |
| Issue | 14 nyitott / 180 lezárt |
| `TODO`/`FIXME` a kódban | 0 |

## Erősségek

- **Átgondolt rétegzés**: UI → services → db / detectors / embeddings; a nagy Qt osztályok témánkénti mixinekre bontva (`main_window/`, `panels/image_browser/`).
- **Robusztus fallback lánc**: Coral → YuNet → SSD → Haar; hiányzó opcionális függőség nem okoz összeomlást (InsightFace, gdrive, Coral).
- **Erős tesztlefedettség**: kb. 1 sor teszt 3 sor kódra, a UI-ra is vannak smoke tesztek (pytest-qt).
- **Kétnyelvű UI** (EN/HU) egységes `t("key")` API-val.
- **Teljes release-lánc**: három platform, aláírt és notarizált macOS build, Sparkle auto-update, appcast generálás, közösségi poszt automatizálás.
- **Gyors iteráció**: a hibajegyek nagy része napokon belül commitot kap, issue-hivatkozásos commit üzenetekkel.

## Kockázatok, gyenge pontok

### 1. Felismerési minőség — a fő nyitott terület
A nyitott jegyek közel fele a felismerés pontosságáról szól (#4, #73, #135, #136, #146, #161, #163, #173). A rendszer több egymásra épülő heurisztikát használ (detektor-szavazás, validátor, nem-arc szűrő, beágyazott doboz szűrés, MLP-ensemble). Ez nehezen mérhető és könnyen regresszál.
- **Javaslat**: legyen egy rögzített, címkézett **benchmark adatkészlet** (pl. `scripts/deep_eval.py` alapján) és CI-ban futó pontossági metrika (precision / recall / fals-pozitív arány / unknown arány), hogy a #136/#146-szerű változtatások mérhetők legyenek.

### 2. Platformfüggő viselkedés (Windows)
#173 (rosszabb minőség Windowson) és #80 (hangrögzítés) is a Windows-specifikus környezetből ered: csendes fallback alacsonyabb minőségű modellre, eszköznevek lokalizációja.
- **Javaslat**: induláskor / a diagnosztikában jól látható jelzés, ha az app fallback detektorral vagy embedderrel fut (a „Zero-Research, One-Click Fix” elv szerint egy javító gombbal).

### 3. Modulméret és komplexitás
29 modul haladja meg a ~800 soros irányelvet. A legnagyobbak: `export_service.py` (3 607), `screen_recorder_service.py` (2 283), `settings_dialog.py` (2 084), `local_server_export_service.py` (1 979), `models.py` (1 931), `deep_recognition_service.py` (1 909).
- **Javaslat**: fokozatos szétbontás, elsőként az `export_service` (formátumonként) és a `settings_dialog` (tabonként, ahogy a `gdrive_settings_tab` / `shortcuts_settings_tab` már mutatja).

### 4. Elavult dokumentáció
- A `README.md` „Architecture” fejezete a korai MVP-t írja le: nem létező fájlokra hivatkozik (`app/ui/main_window.py`, `app/workers/pipeline_worker.py`), és 4 táblát említ a mostani 31 helyett.
- Az `AGENT.md` szintén hivatkozik a már nem létező `app/workers/pipeline_worker.py`-ra.
- **Javaslat**: a README architektúra részét a [PROJEKT_FELEPITES.md](PROJEKT_FELEPITES.md) alapján frissíteni.

### 5. Adatbázis-migráció
31 tábla, sűrű sémaváltozás, Alembic nincs a függőségek között. Migráció kézzel a `database.py`-ban.
- **Javaslat**: séma-verzió tábla és migrációs teszt régi DB-fájlokkal (különösen `.facedb` csomag importnál).

### 6. Függőség-korlátok
`numpy<2.0` pin, közben a #197 numpy-belső pickle hiba jelezte, hogy a környezetek eltérnek. A deep modell mentése pickle-alapú.
- **Javaslat**: verziófüggetlen modellformátum (pl. `npz` + JSON metaadat) és numpy 2 kompatibilitás tesztelése.

### 7. Repo higiénia
A munkakönyvtárban nem követett tartalom: `.agents/`, `.github/agents|prompts|skills/`, `openspec/`, `copilot-setup-steps.yml`, és módosított `config.yaml`. A `config.yaml` helyi beállítás — érdemes `.gitignore`-ba tenni (a minta a `config.example.yaml`).

## Fejlesztési ütem

- 2026-04 → 06: gyors funkcióbővítés (alap pipeline, export, családfa, felvétel, Drive).
- 2026-06 → 10: hangsúly a felismerési minőségen és hibajavításon (#136: 10 commit, #146: 6 commit).
- Utolsó commit: 2026-10-13; azóta nincs aktivitás (~6 hét).

## Javasolt prioritások

1. Nyitott jegyek lezárása, ahol a javítás már commitolva van (#197, #196, #161) — felhasználói ellenőrzés után.
2. Felismerési benchmark + metrika (#136, #146, #163, #135 közös alapja).
3. Windows fallback láthatóvá tétele (#173).
4. Dokumentáció frissítése (README, AGENT.md).
5. `export_service` és `settings_dialog` szétbontása.
