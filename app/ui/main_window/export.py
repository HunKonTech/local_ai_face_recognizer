"""Collage, export, project/database package import-export, path repair."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from app.config import save_db_path
from app.db.database import ensure_unknown_person, init_db, session_scope
from app.db.models import Person
from app.ui.dialogs.export_dialog import ExportDialog
from app.ui.i18n import t

from .helpers import (
    _last_dir,
    _save_dir,
)

log = logging.getLogger(__name__)


class ExportMixin:
    """Collage, export, project/database package import-export, path repair."""

    @Slot()
    def _on_export_collage_html(self) -> None:
        """Export all collages to a static HTML gallery."""
        target = QFileDialog.getExistingDirectory(
            self,
            t("html_export_folder"),
            _last_dir("paths/last_export", str(Path.home())),
        )
        if not target:
            return
        _save_dir("paths/last_export", target)

        def work(ctx):  # noqa: ANN001 — runs on a worker thread
            from app.services.export_service import ExportService
            with session_scope() as session:
                return ExportService(session).export_collage_html(
                    target,
                    progress_cb=ctx.report,
                    cancel_token=ctx.token,
                )

        def on_done(out: object) -> None:
            QMessageBox.information(
                self, t("collage_html_export"),
                t("static_site_ready", path=out)
            )

        def on_error(message: str) -> None:
            QMessageBox.critical(self, t("export_error"), message)

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_html_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=on_done,
            on_error=on_error,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @Slot()
    def _on_open_export(self) -> None:
        person_name: Optional[str] = None
        if self._current_person_id is not None:
            with session_scope() as session:
                p = session.get(Person, self._current_person_id)
                if p:
                    person_name = p.name
        dlg = ExportDialog(
            current_person_id=self._current_person_id,
            current_person_name=person_name,
            current_image_id=self._preview_panel.current_image_id,
            on_collage_import=self._on_import_collage,
            on_collage_html_export=self._on_export_collage_html,
            on_project_export=self._on_export_project_package,
            on_project_import=self._on_import_project_package,
            on_db_export=self._on_export_database_package,
            on_db_import=self._on_import_database_package,
            on_path_repair=self._open_path_repair_dialog,
            on_deep_model_export=self._on_export_deep_model,
            on_deep_model_import=self._on_import_deep_model,
            parent=self,
        )
        dlg.exec()

    # ------------------------------------------------------------------
    # Full project package (.facepack) export / import
    # ------------------------------------------------------------------

    def _build_package_service(self) -> "ProjectPackageService":
        from app.services.image_library_service import get_image_library_optional
        from app.services.project_package_service import ProjectPackageService

        lib = get_image_library_optional()
        return ProjectPackageService(
            db_path=self._db_path,
            crops_dir=self._config.crops_dir_resolved,
            library_root=lib.library_root if lib else None,
            local_config_path=lib.local_config_path() if lib else None,
        )

    @Slot()
    def _on_export_project_package(self) -> None:
        from app.services.project_package_service import PACKAGE_EXTENSION

        default_name = f"face_local_project{PACKAGE_EXTENSION}"
        start_dir = _last_dir("paths/last_package", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("pkg_export_dialog"),
            str(Path(start_dir) / default_name),
            t("pkg_filter"),
        )
        if not path:
            return
        _save_dir("paths/last_package", str(Path(path).parent))

        try:
            from app import __version__ as app_version
        except Exception:  # noqa: BLE001
            app_version = ""

        svc = self._build_package_service()

        def work(ctx):  # noqa: ANN001 — worker thread
            def on_progress(cur: int, total: int, msg: str) -> None:
                ctx.checkpoint()
                pct = int(cur / total * 100) if total else 0
                ctx.report(min(pct, 100), msg or "")

            return svc.export_package(
                path, app_version=str(app_version), progress_cb=on_progress
            )

        def on_done(result: object) -> None:
            msg = t(
                "pkg_export_ok",
                images=result.image_count,
                crops=result.crop_count,
                path=result.package_path,
            )
            if result.warning_count:
                msg += t("pkg_export_warn", missing=result.warning_count)
            QMessageBox.information(self, t("pkg_export_done"), msg)

        def on_error(message: str) -> None:
            QMessageBox.critical(self, t("pkg_error_title"), message)

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_pkg_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=on_done,
            on_error=on_error,
        )
        self._on_open_task_manager()

    @Slot()
    def _on_import_project_package(self) -> None:
        from app.services.project_package_service import ProjectPackageService

        path, _ = QFileDialog.getOpenFileName(
            self,
            t("pkg_import_dialog"),
            _last_dir("paths/last_package", str(Path.home())),
            t("pkg_filter"),
        )
        if not path:
            return
        _save_dir("paths/last_package", str(Path(path).parent))

        validation = ProjectPackageService.validate_package(path)
        if not validation.ok:
            QMessageBox.critical(
                self, t("pkg_error_title"), "\n".join(validation.errors)
            )
            return

        dest = QFileDialog.getExistingDirectory(
            self, t("pkg_import_dest"), str(Path.home())
        )
        if not dest:
            return
        # Extract into a dedicated, non-empty-safe subfolder of the chosen dir.
        dest_dir = Path(dest) / Path(path).stem
        if dest_dir.exists() and any(dest_dir.iterdir()):
            from datetime import datetime
            dest_dir = Path(dest) / f"{Path(path).stem}_{datetime.now():%Y%m%d_%H%M%S}"

        reply = QMessageBox.question(
            self,
            t("pkg_import_title"),
            t("pkg_import_confirm", dest=dest_dir),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        def work(ctx):  # noqa: ANN001 — worker thread
            def on_progress(cur: int, total: int, msg: str) -> None:
                ctx.checkpoint()
                pct = int(cur / total * 100) if total else 0
                ctx.report(min(pct, 100), msg or "")

            return ProjectPackageService.import_package(
                path, dest_dir, progress_cb=on_progress
            )

        def on_done(result: object) -> None:
            reply = QMessageBox.question(
                self,
                t("pkg_import_title"),
                t("pkg_import_ok", dest=result.project_dir),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._activate_imported_project(result)
            else:
                QMessageBox.information(
                    self, t("pkg_import_title"),
                    t("pkg_import_later", dest=result.project_dir),
                )

        def on_error(message: str) -> None:
            QMessageBox.critical(self, t("pkg_error_title"), message)

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_pkg_import"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=on_done,
            on_error=on_error,
        )
        self._on_open_task_manager()

    def _activate_imported_project(self, result) -> None:
        """Open the freshly imported project as the active database."""
        from app.services.image_library_service import get_image_library_optional
        from app.services.project_package_service import ProjectPackageService

        new_db = str(result.db_path)
        self._db_path = new_db
        self._config.storage.db_path = new_db
        self._config.storage.crops_dir = str(result.crops_dir)
        save_db_path(new_db)
        init_db(new_db)  # also re-initialises the ImageLibraryService
        ensure_unknown_person()

        # Point the library root at the extracted images and remap crop paths,
        # then migrate any legacy absolute paths to portable relative paths.
        lib = get_image_library_optional()
        if lib is not None:
            try:
                lib.set_library_root(result.images_dir)
            except OSError:
                log.warning("Imported images dir missing: %s", result.images_dir)
        with session_scope() as session:
            ProjectPackageService.remap_crop_paths(session, result.crops_dir)
            # Repoint every bundled image at its extracted copy so records with
            # only an original-machine absolute path (no portable relative_path)
            # still resolve here. This must run before the migration fallback,
            # which can only handle paths already under the new root.
            ProjectPackageService.remap_image_paths(
                session, result.manifest, result.images_dir
            )
            if lib is not None and lib.is_available():
                try:
                    lib.migrate_to_relative_paths(
                        session, result.images_dir, validate_files=False
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Relative-path migration after import failed")
            missing_originals = self._count_unresolved_images(session)

        # Update the scan source folders to point at the extracted images dir so
        # re-scanning works out of the box without manual reconfiguration.
        images_dir_str = str(result.images_dir.resolve())
        self._root_folders = [images_dir_str]
        self._save_folders([images_dir_str])
        self._set_folder_label([images_dir_str])
        self._scan_modes_btn.setEnabled(True)

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        self._image_browser.refresh()
        if hasattr(self, "_locations_panel"):
            self._locations_panel.refresh()
        if hasattr(self, "_persons_panel"):
            self._persons_panel.refresh()
        if hasattr(self, "_objects_panel"):
            self._objects_panel.refresh()
        msg = t("pkg_import_opened")
        if missing_originals:
            msg += t("pkg_import_missing", missing=missing_originals)
            log.warning(
                "%d imported image(s) could not be located on disk",
                missing_originals,
            )
        QMessageBox.information(self, t("pkg_import_title"), msg)
        log.info("Activated imported project: %s", new_db)

    @staticmethod
    def _count_unresolved_images(session) -> int:
        """Count images whose original file cannot be found after import.

        Resolves each :class:`Image` against the (now reconfigured) library
        root and counts those whose file is absent on disk, so the user is told
        plainly when some originals did not survive the round-trip instead of
        only discovering it later as a broken preview.
        """
        import os

        from app.db.models import Image
        from app.services.image_library_service import resolve_image_path

        missing = 0
        for image in session.query(Image).all():
            try:
                resolved = resolve_image_path(image)
            except Exception:  # noqa: BLE001
                resolved = None
            if resolved is None:
                missing += 1
                continue
            try:
                if not os.path.exists(str(resolved)):
                    missing += 1
            except OSError:
                missing += 1
        return missing

    # ------------------------------------------------------------------
    # Database-only package (.facedb) export / import
    # ------------------------------------------------------------------

    @Slot()
    def _on_export_database_package(self) -> None:
        """Export the database alone, with library-root-relative image paths."""
        from app.services.database_package_service import (
            PACKAGE_EXTENSION,
            DatabasePackageService,
        )
        from app.services.image_library_service import get_image_library_optional

        lib = get_image_library_optional()
        library_root = lib.library_root if lib else None
        if library_root is None:
            # Without a root nothing can be made relative — say so plainly and
            # let the user decide instead of silently exporting absolute paths.
            reply = QMessageBox.question(
                self,
                t("dbpkg_export_done"),
                t("dbpkg_export_no_root"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        default_name = f"face_local_database{PACKAGE_EXTENSION}"
        start_dir = _last_dir("paths/last_package", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("dbpkg_export_dialog"),
            str(Path(start_dir) / default_name),
            t("dbpkg_filter"),
        )
        if not path:
            return
        _save_dir("paths/last_package", str(Path(path).parent))

        try:
            from app import __version__ as app_version
        except Exception:  # noqa: BLE001
            app_version = ""

        svc = DatabasePackageService(
            db_path=self._db_path, library_root=library_root
        )

        def work(ctx):  # noqa: ANN001 — worker thread
            def on_progress(cur: int, total: int, msg: str) -> None:
                ctx.checkpoint()
                pct = int(cur / total * 100) if total else 0
                ctx.report(min(pct, 100), msg or "")

            return svc.export_database(
                path, app_version=str(app_version), progress_cb=on_progress
            )

        def on_done(result: object) -> None:
            msg = t(
                "dbpkg_export_ok",
                images=result.image_count,
                path=result.package_path,
            )
            if result.has_warnings:
                msg += t("dbpkg_export_warn", outside=result.outside_root_count)
            QMessageBox.information(self, t("dbpkg_export_done"), msg)

        def on_error(message: str) -> None:
            QMessageBox.critical(self, t("dbpkg_error_title"), message)

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_dbpkg_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=on_done,
            on_error=on_error,
        )
        self._on_open_task_manager()

    @Slot()
    def _on_import_database_package(self) -> None:
        """Import a .facedb and re-base its paths onto a local base folder."""
        from app.services.database_package_service import DatabasePackageService
        from app.services.image_library_service import get_image_library_optional

        path, _ = QFileDialog.getOpenFileName(
            self,
            t("dbpkg_import_dialog"),
            _last_dir("paths/last_package", str(Path.home())),
            t("dbpkg_filter"),
        )
        if not path:
            return
        _save_dir("paths/last_package", str(Path(path).parent))

        validation = DatabasePackageService.validate_package(path)
        if not validation.ok:
            QMessageBox.critical(
                self, t("dbpkg_error_title"), "\n".join(validation.errors)
            )
            return

        # The base image folder on THIS machine — the local counterpart of the
        # exporting machine's library root.
        lib = get_image_library_optional()
        current_root = lib.library_root if lib else None
        root = QFileDialog.getExistingDirectory(
            self,
            t("dbpkg_import_root"),
            str(current_root) if current_root else str(Path.home()),
        )
        if not root:
            return

        dest = QFileDialog.getExistingDirectory(
            self, t("dbpkg_import_dest"), str(Path.home())
        )
        if not dest:
            return
        dest_dir = Path(dest) / Path(path).stem
        if dest_dir.exists() and any(dest_dir.iterdir()):
            from datetime import datetime
            dest_dir = Path(dest) / f"{Path(path).stem}_{datetime.now():%Y%m%d_%H%M%S}"

        reply = QMessageBox.question(
            self,
            t("dbpkg_import_title"),
            t("dbpkg_import_confirm", dest=dest_dir, root=root),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        def work(ctx):  # noqa: ANN001 — worker thread
            def on_progress(cur: int, total: int, msg: str) -> None:
                ctx.checkpoint()
                pct = int(cur / total * 100) if total else 0
                ctx.report(min(pct, 100), msg or "")

            return DatabasePackageService.import_database(
                path, dest_dir, root, progress_cb=on_progress
            )

        def on_done(result: object) -> None:
            reply = QMessageBox.question(
                self,
                t("dbpkg_import_title"),
                t(
                    "dbpkg_import_ok",
                    dest=result.project_dir,
                    resolved=result.resolved,
                    unresolved=result.unresolved,
                ),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._activate_imported_database(result)
            else:
                QMessageBox.information(
                    self, t("dbpkg_import_title"),
                    t("dbpkg_import_later", dest=result.project_dir),
                )

        def on_error(message: str) -> None:
            QMessageBox.critical(self, t("dbpkg_error_title"), message)

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_dbpkg_import"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=on_done,
            on_error=on_error,
        )
        self._on_open_task_manager()

    def _activate_imported_database(self, result) -> None:
        """Open the freshly imported database as the active project."""
        from app.services.image_library_service import get_image_library_optional

        new_db = str(result.db_path)
        self._db_path = new_db
        self._config.storage.db_path = new_db
        self._config.storage.crops_dir = str(result.crops_dir)
        save_db_path(new_db)
        init_db(new_db)  # also re-initialises the ImageLibraryService
        ensure_unknown_person()

        lib = get_image_library_optional()
        if lib is not None:
            try:
                lib.set_library_root(result.library_root)
            except OSError:
                log.warning("Imported library root missing: %s", result.library_root)

        # Scan sources point at the chosen base folder so a re-scan works
        # without any further configuration.
        root_str = str(Path(result.library_root))
        self._root_folders = [root_str]
        self._save_folders([root_str])
        self._set_folder_label([root_str])
        self._scan_modes_btn.setEnabled(True)

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        self._image_browser.refresh()
        for panel in ("_locations_panel", "_persons_panel", "_objects_panel"):
            if hasattr(self, panel):
                getattr(self, panel).refresh()

        # The package carries no crops, so every crop reference was cleared on
        # import.  Rebuild them from the originals right away instead of waiting
        # for the next app start (LOW priority — the user can keep working).
        self._submit_crop_repair_task()

        log.info(
            "Activated imported database: %s (%d unresolved image(s))",
            new_db, result.unresolved,
        )

        if result.unresolved:
            reply = QMessageBox.question(
                self,
                t("dbpkg_import_title"),
                t("dbpkg_import_opened")
                + t("dbpkg_import_unresolved", n=result.unresolved),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._open_path_repair_dialog()
            return
        QMessageBox.information(
            self, t("dbpkg_import_title"), t("dbpkg_import_opened")
        )

    # ------------------------------------------------------------------
    # Missing-image path repair (scan & match)
    # ------------------------------------------------------------------

    @Slot()
    def _open_path_repair_dialog(self) -> None:
        """Open the review window for records pointing at missing files."""
        from app.services.image_library_service import get_image_library_optional
        from app.ui.dialogs.image_path_repair_dialog import ImagePathRepairDialog

        roots: list[str] = []
        lib = get_image_library_optional()
        if lib is not None and lib.library_root is not None:
            roots.append(str(lib.library_root))
        for folder in self._root_folders or []:
            if folder not in roots:
                roots.append(folder)

        existing = getattr(self, "_path_repair_dialog", None)
        if existing is not None:
            existing.close()

        def on_applied(_n: int) -> None:
            self._refresh_persons()
            self._image_browser.refresh()
            # Newly re-attached originals can finally produce their face crops.
            self._submit_crop_repair_task()

        # Kept on the window so the non-modal dialog is not garbage-collected
        # while its background task is still running.
        self._path_repair_dialog = ImagePathRepairDialog(
            search_roots=roots,
            extensions=self._config.scan.image_extensions,
            on_applied=on_applied,
            parent=self,
        )
        self._path_repair_dialog.show()
        self._path_repair_dialog.raise_()
