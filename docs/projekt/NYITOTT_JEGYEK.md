# Nyitott jegyek

> Forrás: GitHub `HunKonTech/local_ai_face_recognizer`, lekérve 2026-11-27 · **14 nyitott**, 180 lezárt issue, nyitott PR nincs.
> A „Commitok” oszlop a `#szám` hivatkozású commitokat számolja a git logban.

## Összesítő

| # | Cím | Címkék | Létrehozva | Frissítve | Commitok | Állapot |
|---|---|---|---|---|---:|---|
| [#197](https://github.com/HunKonTech/local_ai_face_recognizer/issues/197) | AI tanítás hiba: Can't pickle… `numpy._core.multiarray` | – | 10-11 | 10-12 | 1 | ✅ javítva (`8346be8`), lezárásra vár |
| [#196](https://github.com/HunKonTech/local_ai_face_recognizer/issues/196) | Elmozgatott/átnevezett képek | – | 10-11 | 10-15 | 1 | 🟡 javítva (`817a465`), nyitott kérdés a kommentben |
| [#173](https://github.com/HunKonTech/local_ai_face_recognizer/issues/173) | Windows rossz minőségű felismerés | bug, status:red | 06-25 | 10-11 | 3 | 🔴 ok ismeretlen |
| [#168](https://github.com/HunKonTech/local_ai_face_recognizer/issues/168) | Web export frissítés | bug, status:red | 06-23 | 10-12 | 5 | 🔴 szerver-stabilitás / infra |
| [#163](https://github.com/HunKonTech/local_ai_face_recognizer/issues/163) | A klasszikus még mindig sok fals találatot ad | status:backlog | 06-20 | 09-18 | 0 | ⚪ backlog |
| [#161](https://github.com/HunKonTech/local_ai_face_recognizer/issues/161) | Korábban kategorizált arcra több unknown-t ismer fel | bug, status:red | 06-20 | 10-11 | 1 | 🟡 javítva (`1c9dec4`), ellenőrzésre vár |
| [#150](https://github.com/HunKonTech/local_ai_face_recognizer/issues/150) | Arcok áthelyezése elromlott (csak egyet helyez át) | status:question | 06-12 | 09-18 | 0 | ❓ pontosítás kell |
| [#146](https://github.com/HunKonTech/local_ai_face_recognizer/issues/146) | Sok fals arc felismerés | bug, status:red | 06-11 | 09-18 | 6 | 🔴 folyamatban |
| [#136](https://github.com/HunKonTech/local_ai_face_recognizer/issues/136) | Személy csoportosítás DeepLearning | bug, status:red | 06-10 | 10-13 | 10 | 🔴 folyamatban (utolsó: `0b20af6`) |
| [#135](https://github.com/HunKonTech/local_ai_face_recognizer/issues/135) | Arc felismerés javítás | status:backlog | 06-09 | 09-18 | 2 | ⚪ backlog, gyűjtőjegy |
| [#111](https://github.com/HunKonTech/local_ai_face_recognizer/issues/111) | EXIF adatok közé mentés | enhancement, status:backlog | 06-06 | 10-11 | 4 | 🟡 részben kész (`a4bec62`) |
| [#80](https://github.com/HunKonTech/local_ai_face_recognizer/issues/80) | Nem rögzít hangot | bug, status:red | 06-02 | 10-11 | 3 | 🔴 Windows loopback |
| [#73](https://github.com/HunKonTech/local_ai_face_recognizer/issues/73) | Egyező képeket nem találja meg | status:backlog | 05-31 | 10-11 | 0 | ⚪ backlog |
| [#4](https://github.com/HunKonTech/local_ai_face_recognizer/issues/4) | Egy személyhez rendeli az összes ismeretlent | status:backlog | 05-22 | 10-11 | 0 | ⚪ backlog, legrégebbi |

Dátumok: 2026. · Állapot: saját értékelés az issue szövege, kommentek és commitok alapján.

## Csoportosítás téma szerint

### Felismerési minőség (8)
- **#136** Személy csoportosítás DeepLearning — MLP-ensemble motor kész; maradék panasz: kézzel megadott arcok gyenge megerősítése, nem-arcok (fél arc, szőnyegminta) felismerése. Utolsó komment: AI újraolvasás / újraépítés nulláról szükséges a régi adatokon.
- **#146** Sok fals arc felismerés — 6 commit, összefügg #136-tal.
- **#163** Klasszikus (nem deep) mód sok fals találat.
- **#161** Több unknown egy már kategorizált arcon — beágyazott doboz szűrés commitolva, felhasználói ellenőrzés kell.
- **#173** Windows-on rosszabb minőség — gyanú: csendes fallback, mert a háttérben nem települnek a csomagok.
- **#135** Arc felismerés javítás — általános gyűjtőjegy.
- **#4** Minden ismeretlent egy személyhez rendel (törlés + scan után is).
- **#73** Egyező (duplikált / deoldified) képek nem kerülnek párosításra különböző mappákban.

### Képkezelés / adat (3)
- **#196** Elmozgatott, átnevezett képek — tartalom-hash alapú újrakapcsolás kész; nyitott kérdés: hogyan lehet a hiányzó képeket újratölteni.
- **#111** Felismert személyek mentése EXIF-be (név, ID, téglalap, megjegyzés) — részben kész.
- **#150** Több kijelölt arc áthelyezése csak egyet mozgat.

### AI tanítás (1)
- **#197** Pickle hiba a deep modell mentésekor — javítva, atomikus mentéssel.

### Export / infrastruktúra (1)
- **#168** Web export frissítés — a `csaladfa.kncsk.hu` időnként leáll; javaslat: Linux / Docker szerver.

### Képernyőfelvétel (1)
- **#80** Nincs hang — Windowson az ffmpeg dshow csak bemeneti eszközt rögzít, rendszerhanghoz loopback eszköz (Stereo Mix) kell; a lokalizált eszköznév felismerése javítva.

## Gyors teendők

1. **Lezárható ellenőrzés után**: #197, #161, #196 (a #196-on a kommentben feltett kérdést előbb meg kell válaszolni).
2. **Pontosítás**: #150 (`status:question`) — reprodukálási lépések.
3. **Összevonható**: #135 → #136 / #146 (átfedő felismerési jegyek).
4. **Hiányzó címke**: #197 és #196 nem kapott címkét.
