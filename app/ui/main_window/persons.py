"""Person rename/merge/delete, batch face moves, person list refresh."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Face, Person
from app.services.identity_service import BulkReassignResult, IdentityService
from app.ui.dialogs.merge_dialog import MergeDialog
from app.ui.dialogs.move_faces_dialog import MoveFacesDialog
from app.ui.dialogs.person_info_dialog import PersonInfoDialog
from app.ui.i18n import t

from .helpers import (
    _last_dir,
    _save_dir,
)

log = logging.getLogger(__name__)


class PersonsMixin:
    """Person rename/merge/delete, batch face moves, person list refresh."""

    @Slot()
    def _on_merge(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            source = session.get(Person, self._current_person_id)
            if source is None or len(persons) < 2:
                return

            # One aggregate COUNT query for the "name (N)" labels — loading
            # every person's faces just to count them pulled the whole face
            # table (embeddings included) into RAM on a large database.
            from sqlalchemy import func
            face_counts = dict(
                session.query(Face.person_id, func.count(Face.id))
                .filter(Face.person_id.isnot(None))
                .group_by(Face.person_id)
                .all()
            )

            # Match scores are computed in the background: the dialog opens
            # instantly with a "computing…" hint (shared selector behaviour).
            recognition_cfg = getattr(self._config, "recognition", None) if self._config else None
            dlg = MergeDialog(
                source,
                persons,
                parent=self,
                face_counts=face_counts,
                score_person_id=source.id,
                recognition_cfg=recognition_cfg,
                config=self._config,
            )
            if dlg.exec() != MergeDialog.Accepted:
                return
            target_id = dlg.target_person_id()
            if target_id is None:
                return

            try:
                IdentityService(session).merge_persons(
                    source_id=self._current_person_id, target_id=target_id
                )
            except ValueError as exc:
                QMessageBox.warning(self, t("merge_error_title"), str(exc))
                return

        self._current_person_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()

    @Slot()
    def _on_delete_person(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            if person.is_protected:
                QMessageBox.warning(self, t("protected_delete_title"), t("protected_delete_msg"))
                return
            name = person.name
            n_faces = len(person.faces)
            # The "exclude forever" option is embedding-based and only meaningful
            # for auto-named "Unknown N" identities — never offer it for a named
            # person, whose embeddings must not land on the ignore list.
            can_ignore = person.is_auto_named

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(t("delete_person_title"))
        box.setText(t("delete_person_confirm", name=name, n=n_faces))
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        ignore_check: Optional[QCheckBox] = None
        if can_ignore:
            ignore_check = QCheckBox(t("delete_person_ignore_check"))
            ignore_check.setToolTip(t("ignore_person_tip"))
            box.setCheckBox(ignore_check)
        if box.exec() != QMessageBox.Yes:
            return
        ignore_embeddings = ignore_check is not None and ignore_check.isChecked()

        try:
            with session_scope() as session:
                result = IdentityService(session).delete_person(
                    self._current_person_id,
                    remove_faces=True,
                    ignore_embeddings=ignore_embeddings,
                )
        except ValueError as exc:
            QMessageBox.warning(self, t("delete_person_title"), str(exc))
            return

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._delete_person_btn.setEnabled(False)
        self._rename_btn.setEnabled(False)
        self._merge_btn.setEnabled(False)
        self._person_info_btn.setEnabled(False)
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        if result.n_ignored:
            self._status_label.setText(
                t("delete_person_status_ignored", name=name,
                  n=result.n_faces, k=result.n_ignored)
            )
        else:
            self._status_label.setText(
                t("delete_person_status", name=name, n=result.n_faces)
            )
        log.info(
            "Person '%s' deleted: %d face(s) removed, %d crop file(s) unlinked, "
            "%d embedding(s) ignored.",
            name, result.n_faces, result.n_crops_removed, result.n_ignored,
        )

    @Slot()
    def _on_manage_ignored_faces(self) -> None:
        """Open the manager for the permanent ignore list."""
        from app.ui.dialogs.ignored_faces_dialog import IgnoredFacesDialog
        dlg = IgnoredFacesDialog(config=self._config, parent=self)
        dlg.exec()
        if dlg.changed():
            self._refresh_persons()
            self._image_browser._reload_current_face_data()

    @Slot()
    def _on_remove_face(self) -> None:
        if self._current_face_id is None or self._current_person_id is None:
            return

        reply = QMessageBox.question(
            self,
            t("remove_face_title"),
            t("remove_face_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            IdentityService(session).remove_face_from_cluster(self._current_face_id)

        self._on_person_selected(self._current_person_id)

    @Slot()
    def _on_reassign_face(self) -> None:
        if self._current_face_id is None:
            return

        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()

            class _FakePerson:
                name = f"Face #{self._current_face_id}"
                id = -1
                faces: list = []

            # One aggregate COUNT query for the "name (N)" labels instead of
            # lazy-loading every person's faces (huge on a large database).
            from sqlalchemy import func
            face_counts = dict(
                session.query(Face.person_id, func.count(Face.id))
                .filter(Face.person_id.isnot(None))
                .group_by(Face.person_id)
                .all()
            )

            # The face being reassigned is scored in the background so the
            # dialog opens instantly; percentages appear when scoring finishes.
            recognition_cfg = getattr(self._config, "recognition", None) if self._config else None
            dlg = MergeDialog(
                _FakePerson(),
                persons,
                parent=self,
                face_counts=face_counts,
                score_face_ids=[self._current_face_id],
                recognition_cfg=recognition_cfg,
                config=self._config,
            )
            dlg.setWindowTitle(t("reassign_title"))
            if dlg.exec() != MergeDialog.Accepted:
                return
            target_id = dlg.target_person_id()
            if target_id is None:
                return
            face = session.get(Face, self._current_face_id)
            log.debug(
                "Reassign face: face_id=%d crop=%s bbox=(%s,%s,%s,%s) "
                "from person_id=%s → target_person_id=%d",
                self._current_face_id,
                face.crop_path if face else "?",
                face.bbox_x if face else "?", face.bbox_y if face else "?",
                face.bbox_w if face else "?", face.bbox_h if face else "?",
                self._current_person_id, target_id,
            )
            IdentityService(session).reassign_face(self._current_face_id, target_id)
            log.info(
                "Reassign done: face_id=%d → person_id=%d",
                self._current_face_id, target_id,
            )

        if self._current_person_id:
            self._on_person_selected(self._current_person_id)
        self._show_face_in_preview(self._current_face_id)
        self._image_browser._reload_current_face_data()

    # ------------------------------------------------------------------
    # Batch face move (multi-select in the cluster grid)
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_face_selection_changed(self, count: int) -> None:
        if count <= 0:
            self._sel_count_lbl.setText("")
        else:
            self._sel_count_lbl.setText(t("faces_selected_n", n=count))
        self._move_faces_btn.setVisible(count > 0)

    @Slot()
    def _on_move_faces_batch(self) -> None:
        face_ids = self._cluster_panel.selected_face_ids()
        if not face_ids:
            return

        src_name = ""
        with session_scope() as session:
            if self._current_person_id is not None:
                src = session.get(Person, self._current_person_id)
                src_name = src.name if src is not None else ""
            persons = session.query(Person).order_by(Person.name).all()
            recognition_cfg = getattr(self._config, "recognition", None) if self._config else None
            # Scores are computed in the background: the dialog opens instantly
            # with a "computing…" hint instead of freezing on a large database.
            dlg = MoveFacesDialog(
                face_count=len(face_ids),
                persons=persons,
                exclude_person_id=self._current_person_id,
                score_face_ids=face_ids,
                recognition_cfg=recognition_cfg,
                config=self._config,
                parent=self,
            )
        if dlg.exec() != MoveFacesDialog.Accepted:
            return

        target_id = dlg.selected_person_id()
        new_name = dlg.new_person_name()

        if new_name:
            dst_name = new_name
        elif target_id is not None:
            if target_id == self._current_person_id:
                QMessageBox.information(
                    self, t("move_faces_title"), t("move_faces_same_person")
                )
                return
            with session_scope() as session:
                target = session.get(Person, target_id)
                dst_name = target.name if target is not None else ""
        else:
            return

        reply = QMessageBox.question(
            self,
            t("move_faces_confirm_title"),
            t("move_faces_confirm_msg", n=len(face_ids), src=src_name, dst=dst_name),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                svc = IdentityService(session)
                if new_name:
                    person = Person(name=new_name, is_auto_named=False)
                    session.add(person)
                    session.flush()
                    target_id = person.id
                result = svc.reassign_faces_bulk(face_ids, target_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Bulk face move failed")
            QMessageBox.critical(
                self, t("move_faces_error_title"), t("persons_save_error", error=str(exc))
            )
            return

        self._cluster_panel.clear_selection()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

        # Offer an undo via a dialog button.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(t("move_faces_title"))
        box.setText(t("move_faces_done", n=result.moved_count, dst=dst_name))
        undo_btn = box.addButton(t("move_faces_undo"), QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is undo_btn:
            self._undo_face_move(result)

    def _undo_face_move(self, result: BulkReassignResult) -> None:
        try:
            with session_scope() as session:
                IdentityService(session).restore_face_assignments(
                    result.snapshots, result.removed_persons
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Undo of bulk face move failed")
            QMessageBox.critical(
                self,
                t("move_faces_error_title"),
                t("move_faces_undo_error", error=str(exc)),
            )
            return
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

    @Slot()
    def _on_person_info(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            dlg = PersonInfoDialog(person, parent=self)
            if dlg.exec() != PersonInfoDialog.Accepted:
                return
            person.gender = dlg.gender()
            person.family_code = dlg.family_code() or None
            person.external_family_code = dlg.external_family_code() or None
            person.name_prefix = dlg.name_prefix() or None
            person.last_name = dlg.last_name() or None
            person.first_name = dlg.first_name() or None
            person.second_name = dlg.second_name() or None
            person.nickname = dlg.nickname() or None
            person.married_name = dlg.married_name() or None
            person.birth_place = dlg.birth_place() or None
            person.birth_date = dlg.birth_date() or None
            person.death_date = dlg.death_date() or None
            person.death_place = dlg.death_place() or None
            person.notes = dlg.notes() or None
            session.flush()
            try:
                from app.services.family_service import FamilyService
                FamilyService(session).link_derived_parents(person.id)
            except Exception:
                log.exception("Could not link derived parents for person %d", person.id)

        log.info(
            "Személyadatok mentve: %s %s",
            dlg.last_name(), dlg.first_name()
        )
        # Reflect the saved structured data in the sidebar and the Persons tab
        # table (the dialog only wrote to the DB).
        self._refresh_persons()

    # ------------------------------------------------------------------
    # Collage import / export
    # ------------------------------------------------------------------

    @Slot()
    def _on_import_collage(self) -> None:
        """Import one or more Picasa collage files (.cxf / .cfx)."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            t("open_collage_file"),
            _last_dir("paths/last_collage", str(Path.home())),
            t("picasa_collage_filter"),
        )
        if not files:
            return
        _save_dir("paths/last_collage", str(Path(files[0]).parent))

        # Optionally ask for an extra search root to resolve Windows paths
        search_root = QFileDialog.getExistingDirectory(
            self,
            t("extra_search_root"),
            _last_dir("paths/last_search_root", str(Path.home())),
        )
        if search_root:
            _save_dir("paths/last_search_root", search_root)
        search_roots = [search_root] if search_root else []

        imported, errors = 0, []
        for fpath in files:
            try:
                with session_scope() as session:
                    from app.services.collage_service import CollageService
                    svc = CollageService(session)
                    collage = svc.import_collage(fpath, search_roots=search_roots)
                    log.info(
                        "Kollázs importálva: %s  (%d elem, %d hiányzó)",
                        collage.album_title or fpath,
                        len(collage.nodes),
                        sum(1 for n in collage.nodes if n.src_missing),
                    )
                imported += 1
            except Exception as exc:
                log.error("Kollázs import hiba (%s): %s", fpath, exc)
                errors.append(f"{Path(fpath).name}: {exc}")

        # Switch to collage tab and refresh
        self._collage_panel.refresh_collage_list()
        self._tabs.setCurrentIndex(self._tabs.indexOf(self._collage_panel))

        msg = t("collages_imported", n=imported)
        if errors:
            msg += f"\n\n{t('import_errors')}\n" + "\n".join(errors)
            QMessageBox.warning(self, t("collage_import_title"), msg)
        else:
            self._status_label.setText(msg)
            log.info(msg)

    def _refresh_persons(self) -> None:
        # The sidebar only needs plain values (name, face count, one
        # representative crop per person).  Loading the full Person→Face→Image
        # ORM graph — embedding blobs included — took 15+ seconds at
        # 5000 persons / 100k faces and ran on the UI thread after every
        # single face assignment.  A handful of aggregate queries riding the
        # ix_faces_person_listing covering index do the same work in
        # milliseconds.  Crop-path repair (ensure_unique_face_crops) moved to
        # a one-time startup background task — see _start_crop_repair_task.
        from sqlalchemy import text as _sql

        from app.perf import timed_block
        from app.ui.panels.sidebar_panel import FaceData, SidebarPerson

        with timed_block("ui.refresh_persons"):
            with session_scope() as session:
                person_rows = session.execute(
                    _sql(
                        "SELECT id, name, is_protected, is_auto_named,"
                        "       thumbnail_path "
                        "FROM persons ORDER BY name"
                    )
                ).fetchall()

                count_rows = session.execute(
                    _sql(
                        "SELECT person_id, COUNT(*) FROM faces "
                        "WHERE person_id IS NOT NULL GROUP BY person_id"
                    )
                ).fetchall()
                face_counts = {pid: n for pid, n in count_rows}

                # Best-confidence face id per person (index-only window scan).
                best_rows = session.execute(
                    _sql(
                        "SELECT person_id, face_id FROM ("
                        "  SELECT f.person_id AS person_id, f.id AS face_id,"
                        "         ROW_NUMBER() OVER ("
                        "             PARTITION BY f.person_id"
                        "             ORDER BY f.confidence DESC) AS rn"
                        "  FROM faces f"
                        "  WHERE f.person_id IS NOT NULL"
                        "    AND f.is_excluded = 0"
                        "    AND f.crop_path IS NOT NULL"
                        ") WHERE rn = 1"
                    )
                ).fetchall()
                best_face_id = {pid: fid for pid, fid in best_rows}

                # Faces matching a manually chosen person thumbnail override
                # the best-confidence pick, so the hover popup bbox matches
                # the visible crop.
                thumb_rows = session.execute(
                    _sql(
                        "SELECT f.person_id, f.id FROM faces f "
                        "JOIN persons p ON p.id = f.person_id "
                        "WHERE p.thumbnail_path IS NOT NULL "
                        "  AND f.crop_path = p.thumbnail_path"
                    )
                ).fetchall()
                for pid, fid in thumb_rows:
                    best_face_id[pid] = fid

                face_detail: dict[int, tuple] = {}
                wanted = list(best_face_id.values())
                for start in range(0, len(wanted), 900):
                    chunk = wanted[start:start + 900]
                    placeholders = ",".join(str(int(fid)) for fid in chunk)
                    detail_rows = session.execute(
                        _sql(
                            "SELECT f.id, f.person_id, f.crop_path, i.file_path,"
                            "       f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h "
                            f"FROM faces f JOIN images i ON i.id = f.image_id "
                            f"WHERE f.id IN ({placeholders})"
                        )
                    ).fetchall()
                    for fid, pid, crop, img, bx, by, bw, bh in detail_rows:
                        face_detail[pid] = (fid, crop, img, (bx, by, bw, bh))

            persons: list[SidebarPerson] = []
            for pid, name, is_protected, is_auto_named, thumbnail_path in person_rows:
                detail = face_detail.get(pid)
                if detail is not None:
                    fid, crop, img, bbox = detail
                    crop = thumbnail_path or crop
                    fd = FaceData(
                        face_id=fid, person_id=pid,
                        crop_path=crop, image_path=img, bbox=bbox,
                    )
                else:
                    fd = FaceData(
                        face_id=None, person_id=pid,
                        crop_path=thumbnail_path, image_path=None, bbox=None,
                    )
                persons.append(
                    SidebarPerson(
                        id=pid,
                        name=name,
                        is_protected=bool(is_protected),
                        face_count=face_counts.get(pid, 0),
                        face=fd,
                        is_auto_named=bool(is_auto_named),
                    )
                )
            self._sidebar.populate(persons)
        if hasattr(self, "_family_search"):
            self._family_search.refresh()
        # Keep the standalone Persons tab table in sync with edits made
        # anywhere (image browser, sidebar dialog, batch ops).  It reloads
        # immediately when visible, otherwise lazily on next show.
        if hasattr(self, "_persons_panel"):
            self._persons_panel.mark_stale()
        self._refresh_amerge_btn()
        log.debug("Sidebar refreshed: %d person(s)", len(persons))

    @Slot(int)
    def _open_image_from_family_search(self, image_id: int) -> None:
        self._tabs.setCurrentIndex(1)
        self._image_browser.open_image_by_id(image_id)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    @Slot()
    def _on_open_task_manager(self) -> None:
        from app.ui.dialogs.task_manager_dialog import TaskManagerDialog

        if self._task_manager_dialog is None:
            self._task_manager_dialog = TaskManagerDialog(self)
            self._task_manager_dialog.finished.connect(
                lambda *_: setattr(self, "_task_manager_dialog", None)
            )
        self._task_manager_dialog.show()
        self._task_manager_dialog.raise_()
        self._task_manager_dialog.activateWindow()
