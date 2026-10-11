"""Screen recording controls, metadata and browser context events."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Image, Person
from app.ui.i18n import t

log = logging.getLogger(__name__)


class RecordingMixin:
    """Screen recording controls, metadata and browser context events."""

    # ------------------------------------------------------------------
    # Screen recording
    # ------------------------------------------------------------------

    def _on_record_start(self) -> None:
        from app.services.screen_recorder_service import (
            CaptureDevices,
            RecordingDisplayMode,
            RecordingOptions,
            ScreenRecorderService,
            effective_fps,
            probe_devices,
            resolve_capture_region,
            resolve_ffmpeg,
            selected_displays,
        )

        rec_cfg = self._config.recording
        ffmpeg = resolve_ffmpeg(rec_cfg.ffmpeg_path)
        if not ffmpeg:
            QMessageBox.warning(
                self, t("rec_ffmpeg_missing_title"), t("rec_ffmpeg_missing_body")
            )
            return

        # Privacy notice — show unless the user opted out.
        if not self._confirm_recording_privacy():
            return

        # Pick output folder — default to the last-used / configured dir.
        start_dir = rec_cfg.output_dir or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, t("rec_choose_dir"), start_dir
        )
        if not chosen:
            return
        # Remember the choice (config + persisted settings).
        rec_cfg.output_dir = chosen
        self._save_recording_output_dir(chosen)

        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(chosen) / f"recording_{stamp}"

        devices: CaptureDevices = probe_devices(
            ffmpeg,
            want_system_audio=rec_cfg.capture_system_audio,
            mic_name=rec_cfg.audio_input_device,
            system_audio_name=rec_cfg.system_audio_device,
        )
        if not rec_cfg.capture_microphone:
            devices.microphone = None
            devices.microphone_name = None

        # Resolve which monitors / window to capture, and cap the fps for
        # multi-monitor captures.
        from app.ui.display_utils import active_window_bounds, enumerate_displays
        mode = RecordingDisplayMode.from_value(rec_cfg.display_mode)
        displays = enumerate_displays(rec_cfg.windows_dxgi_output_overrides or None)
        # On macOS the avfoundation "Capture screen N" device index is not the
        # monitor ordinal (capture devices come after the cameras), so map each
        # monitor to its real avfoundation index before resolving the region.
        if sys.platform.startswith("darwin") and displays:
            from app.services.screen_recorder_service import probe_screen_indices
            screen_idx = probe_screen_indices(ffmpeg)
            for i, disp in enumerate(displays):
                if i < len(screen_idx):
                    try:
                        disp.av_index = int(screen_idx[i])
                    except (TypeError, ValueError):
                        disp.av_index = None
        region = resolve_capture_region(
            mode,
            displays,
            rec_cfg.selected_display_ids,
            active_window_bounds(self),
        )
        if sys.platform.startswith("win"):
            # ffmpeg.exe ships without a per-monitor DPI manifest, so gdigrab
            # addresses a *virtualized* desktop.  Ask it what it sees once and
            # convert our physical crop into its coordinate space; without this
            # a scaled multi-monitor layout crops off-screen (i.e. black).
            from app.services.screen_recorder_service import (
                gdigrab_coordinate_scale,
                probe_gdigrab_rect,
                scale_region_for_gdigrab,
            )
            from app.services.windows_display_probe import (
                enumerate_physical_monitors,
                virtual_desktop_rect,
            )
            scale = gdigrab_coordinate_scale(
                probe_gdigrab_rect(ffmpeg),
                virtual_desktop_rect(enumerate_physical_monitors()),
            )
            if scale != 1.0:
                log.info("gdigrab coordinate scale calibrated to %.4f", scale)
                region = scale_region_for_gdigrab(region, scale)
        fps = effective_fps(
            rec_cfg.fps,
            mode,
            displays,
            rec_cfg.selected_display_ids,
            auto_reduce=rec_cfg.auto_reduce_fps,
            multi_monitor_cap=rec_cfg.multi_monitor_fps_cap,
        )
        if sys.platform.startswith("darwin") and mode in (
            RecordingDisplayMode.ALL_DISPLAYS,
            RecordingDisplayMode.ACTIVE_WINDOW,
        ):
            log.info(
                "macOS avfoundation cannot capture a window or merge displays; "
                "recording the primary screen for mode %s", mode.value
            )
        options = RecordingOptions(
            fps=fps,
            quality=rec_cfg.quality,
            segment_seconds=rec_cfg.segment_seconds,
            capture_cursor=rec_cfg.capture_cursor,
            mic_volume=rec_cfg.mic_volume,
            system_volume=rec_cfg.system_volume,
            mute_microphone=rec_cfg.mute_microphone,
            mute_system_audio=rec_cfg.mute_system_audio,
            meter_audio=True,  # drive the live VU meter
            windows_backend=rec_cfg.windows_capture_backend,
            windows_dxgi_adapter=rec_cfg.windows_dxgi_adapter,
        )

        self._recorder = ScreenRecorderService(ffmpeg, parent=self)
        self._recorder.state_changed.connect(self._on_recorder_state)
        self._recorder.elapsed_changed.connect(self._on_recorder_elapsed)
        self._recorder.error.connect(self._on_recorder_error)
        self._recorder.audio_level.connect(self._on_recorder_audio_level)
        self._recorder.audio_validated.connect(self._on_recorder_audio_validated)
        self._recorder.video_validated.connect(self._on_recorder_video_validated)
        self._recorder.preflight_result.connect(self._on_recorder_preflight)
        self._recorder.backend_changed.connect(self._on_recorder_backend_changed)

        from app.services.recording_timeline_log import RecordingTimelineLog
        from app.services.subtitle_service import subtitle_path_for_video
        session_dir.mkdir(parents=True, exist_ok=True)
        self._recording_log = RecordingTimelineLog(
            subtitle_path_for_video(session_dir / "recording.mp4"),
            person_prefix=t("rec_person_prefix"),
        )

        # Metadata — written immediately as "crashed" until a clean finalize.
        captured = selected_displays(mode, displays, rec_cfg.selected_display_ids)
        self._recording_metadata = self._make_recording_metadata(
            session_dir, mode, fps, captured, devices
        )

        self._recorder.start(
            session_dir,
            devices,
            options,
            concat_on_stop=rec_cfg.concat_on_stop,
            region=region,
            preflight=rec_cfg.preflight_black_check,
        )

        # With a pre-flight probe the recorder is still in PREFLIGHT here, so
        # the initial timeline entry is written from the state handler instead.

    def _confirm_recording_privacy(self) -> bool:
        """Show the privacy notice; return ``False`` if the user cancels."""
        from app.app_settings import app_qsettings
        qs = app_qsettings()
        if qs.value("recording/privacy_ack", False, type=bool):
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(t("rec_privacy_title"))
        box.setText(t("rec_privacy_body"))
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        dont_ask = QCheckBox(t("rec_privacy_dont_ask"))
        box.setCheckBox(dont_ask)
        if box.exec() != QMessageBox.Ok:
            return False
        if dont_ask.isChecked():
            qs.setValue("recording/privacy_ack", True)
        return True

    def _make_recording_metadata(
        self, session_dir, mode, fps, captured_displays, devices
    ):
        """Build + write the initial (pessimistic) recording metadata."""
        from datetime import datetime, timezone

        from app.services.recording_metadata import (
            METADATA_FILENAME,
            RecordingMetadata,
            RecordingMetadataWriter,
        )
        from app.services.screen_recorder_service import displays_bounding_box

        box = displays_bounding_box(captured_displays)
        meta = RecordingMetadata(
            started_at=datetime.now(timezone.utc).isoformat(),
            platform=sys.platform,
            fps=fps,
            width=box[2] if box else None,
            height=box[3] if box else None,
            display_mode=mode.value,
            monitors=[
                {
                    "id": d.id,
                    "name": d.name,
                    "width": d.width,
                    "height": d.height,
                    "is_primary": d.is_primary,
                    "x": d.x,
                    "y": d.y,
                }
                for d in captured_displays
            ],
            layout={"x": box[0], "y": box[1], "width": box[2], "height": box[3]}
            if box
            else None,
            captured_microphone=devices.microphone is not None,
            captured_system_audio=devices.system_audio is not None,
        )
        writer = RecordingMetadataWriter(session_dir / METADATA_FILENAME, meta)
        writer.begin()
        return writer

    def _on_record_pause_toggle(self) -> None:
        from app.services.screen_recorder_service import RecorderState
        if self._recorder is None:
            return
        if self._recorder.state is RecorderState.RECORDING:
            self._recorder.pause()
        elif self._recorder.state is RecorderState.PAUSED:
            self._recorder.resume()
            self._notify_recording_context()

    def _on_record_stop(self) -> None:
        if self._recorder is None:
            return
        self._last_audio_validation = None
        self._last_video_validation = None
        elapsed = self._recorder.elapsed_seconds
        # stop() validates the final file and emits ``audio_validated``
        # synchronously, so ``_last_audio_validation`` is populated below.
        final = self._recorder.stop()
        if self._recording_log is not None:
            self._recording_log.finalize(elapsed)
            self._recording_log = None
        self._finalize_recording_metadata(elapsed, crashed=False)
        out_dir = self._recorder.output_dir
        self._recorder = None
        target = final or out_dir
        if target is not None:
            body = t("rec_saved_body", path=str(target))
            video = self._last_video_validation
            if video is not None and video.is_black:
                # A black picture makes the whole recording useless, so it wins
                # over the audio warning and names the remedy directly.
                QMessageBox.warning(
                    self,
                    t("rec_black_video_title"),
                    t("rec_black_video_body", path=str(target)),
                )
                return
            validation = self._last_audio_validation
            if validation is not None and not validation.has_audio:
                # Make a silent recording impossible to miss.  The remedy is
                # platform-specific (BlackHole/TCC on macOS vs. Stereo Mix /
                # Windows mic-privacy on Windows), so pick the right guidance.
                warn_key = (
                    "rec_no_audio_warning_windows"
                    if sys.platform.startswith("win")
                    else "rec_no_audio_warning"
                )
                QMessageBox.warning(
                    self,
                    t("rec_saved_title"),
                    f"{body}\n\n⚠ {t(warn_key)}",
                )
            else:
                QMessageBox.information(self, t("rec_saved_title"), body)

    def _on_recorder_state(self, state) -> None:
        from app.services.screen_recorder_service import RecorderState

        self._recording_controls.set_state(state)
        # Capture may begin after an asynchronous pre-flight probe, so the first
        # timeline entry is written on the edge into RECORDING rather than at
        # the end of _on_record_start.
        if state is RecorderState.RECORDING and self._recording_log is not None:
            self._notify_recording_context()

    def _on_recorder_elapsed(self, seconds: int) -> None:
        self._recording_controls.set_elapsed(seconds)

    def _on_recorder_audio_level(self, db: float) -> None:
        self._recording_controls.set_audio_level(db)

    def _on_recorder_audio_validated(self, validation) -> None:
        """Remember the final-mux audio check so stop() can warn if silent."""
        self._last_audio_validation = validation
        if not validation.has_audio:
            log.warning("recording finished with no usable audio track")
            # Persist into the (still-open) crash-safe metadata for post-mortem.
            if self._recording_metadata is not None:
                self._recording_metadata.add_error(validation.summary())

    def _on_recorder_video_validated(self, validation) -> None:
        """Remember the final-mux picture check so stop() can warn if black."""
        self._last_video_validation = validation
        if validation.is_black:
            log.error("recording finished with an all-black video track")
            if self._recording_metadata is not None:
                self._recording_metadata.add_error(validation.summary())

    def _on_recorder_preflight(self, probe) -> None:
        """Log the pre-flight verdict; a black result is recorded for later."""
        log.info("recorder pre-flight: %s", probe.summary())
        if (not probe.ok or probe.is_black) and self._recording_metadata is not None:
            self._recording_metadata.add_error(probe.summary())

    def _on_recorder_backend_changed(self, backend: str) -> None:
        """A black grabber forced a switch — say so in the status bar."""
        log.warning("recording capture backend switched to %s", backend)
        self.statusBar().showMessage(
            t("rec_backend_switched", backend=backend), 8000
        )

    def _on_recorder_error(self, message: str) -> None:
        QMessageBox.critical(self, t("rec_error_title"), message)
        elapsed = self._recorder.elapsed_seconds if self._recorder is not None else 0
        if self._recording_log is not None and self._recorder is not None:
            self._recording_log.finalize(elapsed)
        self._recording_log = None
        if self._recording_metadata is not None:
            self._recording_metadata.add_error(message)
        self._finalize_recording_metadata(elapsed, crashed=True)

    def _finalize_recording_metadata(self, elapsed: float, *, crashed: bool) -> None:
        """Write the closing metadata record (clean stop or crash)."""
        if self._recording_metadata is None:
            return
        try:
            from datetime import datetime, timezone
            self._recording_metadata.finalize(
                datetime.now(timezone.utc).isoformat(),
                float(elapsed),
                crashed=crashed,
            )
        except Exception:  # noqa: BLE001 — never let metadata break shutdown
            log.debug("could not finalize recording metadata", exc_info=True)
        self._recording_metadata = None

    def _connect_display_events(self) -> None:
        """Watch for monitor hot-plug / layout changes (best-effort, no crash).

        A monitor disappearing or resizing mid-recording can break the active
        ffmpeg capture; that surfaces via the recorder's ``error`` signal (which
        finalizes the log + metadata as crashed).  Here we only log the event so
        it is diagnosable and never let a display-layer hiccup raise.
        """
        try:
            app = QApplication.instance()
            if app is None:
                return
            app.screenAdded.connect(self._on_screen_changed)
            app.screenRemoved.connect(self._on_screen_changed)
        except Exception:  # noqa: BLE001
            log.debug("could not connect display events", exc_info=True)

    def _on_screen_changed(self, _screen=None) -> None:
        try:
            from app.services.screen_recorder_service import RecorderState
            recording = (
                self._recorder is not None
                and self._recorder.state in (
                    RecorderState.RECORDING,
                    RecorderState.PAUSED,
                )
            )
            log.info(
                "display configuration changed (recording=%s); "
                "active capture is unaffected unless a captured monitor was lost",
                recording,
            )
        except Exception:  # noqa: BLE001
            log.debug("screen-change handler failed", exc_info=True)

    def _notify_recording_context(
        self,
        image_name: Optional[str] = None,
        track_person: bool = True,
    ) -> None:
        """Record which image/person is active in the timeline log.

        When *image_name* is given it is used directly (e.g. from the image
        browser); otherwise the face-tab preview's current image is resolved.
        *track_person* controls whether the selected person is logged.
        """
        from app.services.screen_recorder_service import RecorderState
        if self._recording_log is None or self._recorder is None:
            return
        if self._recorder.state not in (
            RecorderState.RECORDING,
            RecorderState.PAUSED,
        ):
            return
        person_name: Optional[str] = None
        image_id = (
            None if image_name is not None else self._preview_panel.current_image_id
        )
        try:
            with session_scope() as session:
                if image_id is not None:
                    img = session.get(Image, image_id)
                    if img is not None:
                        image_name = Path(img.file_path).name
                if track_person and self._current_person_id is not None:
                    p = session.get(Person, self._current_person_id)
                    if p is not None:
                        person_name = p.name
        except Exception:  # noqa: BLE001 — logging must never break the UI
            log.debug("recording context lookup failed", exc_info=True)
        self._recording_log.note_active(
            image_name, person_name, self._recorder.elapsed_seconds
        )

    @Slot(int, str)
    def _on_browser_image_displayed(self, _image_id: int, file_path: str) -> None:
        """Image-browser tab opened an image — update the recording timeline.

        A new image clears any face selection (see ``_load_image``), so the
        active person resets to ``None`` until the user clicks a face.
        """
        self._browser_image_name = Path(file_path).name
        self._browser_person_name = None
        self._notify_browser_context()

    @Slot(object)
    def _on_browser_person_changed(self, person_name) -> None:
        """A face was selected / (re)assigned in the image-browser tab."""
        self._browser_person_name = person_name or None
        self._notify_browser_context()

    def _notify_browser_context(self) -> None:
        """Push the image-browser tab's active image + person to the log."""
        if self._recording_log is None or self._recorder is None:
            return
        from app.services.screen_recorder_service import RecorderState
        if self._recorder.state not in (
            RecorderState.RECORDING,
            RecorderState.PAUSED,
        ):
            return
        self._recording_log.note_active(
            self._browser_image_name,
            self._browser_person_name,
            self._recorder.elapsed_seconds,
        )

    def _save_recording_output_dir(self, path: str) -> None:
        try:
            from app.app_settings import app_qsettings
            app_qsettings().setValue("recording/output_dir", path)
        except Exception:  # noqa: BLE001
            log.debug("could not persist recording output dir", exc_info=True)

    def _load_recording_prefs(self) -> None:
        """Overlay persisted recording settings (QSettings) onto the config."""
        try:
            from app.app_settings import app_qsettings
            qs = app_qsettings()
            rec = self._config.recording
            saved_dir = qs.value("recording/output_dir", None, type=str)
            if saved_dir:
                rec.output_dir = saved_dir
            ffmpeg_path = qs.value("recording/ffmpeg_path", None, type=str)
            if ffmpeg_path:
                rec.ffmpeg_path = ffmpeg_path
            rec.quality = qs.value("recording/quality", rec.quality, type=str)
            rec.fps = qs.value("recording/fps", rec.fps, type=int)
            rec.segment_seconds = qs.value(
                "recording/segment_seconds", rec.segment_seconds, type=int
            )
            rec.capture_cursor = qs.value(
                "recording/capture_cursor", rec.capture_cursor, type=bool
            )
            rec.capture_microphone = qs.value(
                "recording/capture_microphone", rec.capture_microphone, type=bool
            )
            rec.capture_system_audio = qs.value(
                "recording/capture_system_audio", rec.capture_system_audio, type=bool
            )
            # Empty string → keep the auto-pick (None), not a literal "".
            rec.audio_input_device = (
                qs.value("recording/audio_input_device", "", type=str) or None
            )
            rec.system_audio_device = (
                qs.value("recording/system_audio_device", "", type=str) or None
            )
            rec.mic_volume = qs.value(
                "recording/mic_volume", rec.mic_volume, type=float
            )
            rec.system_volume = qs.value(
                "recording/system_volume", rec.system_volume, type=float
            )
            rec.mute_microphone = qs.value(
                "recording/mute_microphone", rec.mute_microphone, type=bool
            )
            rec.mute_system_audio = qs.value(
                "recording/mute_system_audio", rec.mute_system_audio, type=bool
            )
            rec.concat_on_stop = qs.value(
                "recording/concat_on_stop", rec.concat_on_stop, type=bool
            )
            rec.display_mode = qs.value(
                "recording/display_mode", rec.display_mode, type=str
            )
            rec.auto_reduce_fps = qs.value(
                "recording/auto_reduce_fps", rec.auto_reduce_fps, type=bool
            )
            rec.windows_capture_backend = qs.value(
                "recording/windows_capture_backend",
                rec.windows_capture_backend,
                type=str,
            )
            rec.preflight_black_check = qs.value(
                "recording/preflight_black_check",
                rec.preflight_black_check,
                type=bool,
            )
            saved_ids = qs.value("recording/selected_display_ids", None)
            if saved_ids is not None:
                # QSettings may round-trip a single string instead of a list.
                if isinstance(saved_ids, str):
                    saved_ids = [saved_ids] if saved_ids else []
                rec.selected_display_ids = [str(x) for x in saved_ids]
        except Exception:  # noqa: BLE001
            log.debug("could not load recording prefs", exc_info=True)

    # ------------------------------------------------------------------
    # Pipeline slots
    # ------------------------------------------------------------------

    @Slot(bool, str)
    def _on_pipeline_finished(self, success: bool, summary: str) -> None:
        self._set_scanning_state(False)
        self._status_label.setText(summary)
        self._progress_bar.setValue(100)
        self._refresh_persons()
        self._image_browser.refresh()
        self._locations_panel.refresh()
        self._persons_panel.refresh()
        self._objects_panel.refresh()
        new_image_ids = self._pending_object_scan_ids
        self._pending_object_scan_ids = []
        if not success:
            QMessageBox.warning(self, t("warning"), summary)
            return

        self._maybe_search_objects_after_scan(new_image_ids)

        if self._pending_auto_assignment_count > 0:
            reply = QMessageBox.question(
                self,
                t("auto_assignments_found_title"),
                t(
                    "auto_assignments_found_msg",
                    n=self._pending_auto_assignment_count,
                ),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._on_show_suggestions(open_auto_tab=True)
        self._pending_auto_assignment_count = 0
