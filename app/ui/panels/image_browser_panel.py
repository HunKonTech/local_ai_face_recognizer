"""Image Browser Panel — Finder-style tree + preview.

Left   : QTreeWidget — folders expandable, images shown with thumbnail icon + filename
Right  : Full image preview with bbox interaction + face-assign panel

Public API consumed by MainWindow
----------------------------------
    panel = ImageBrowserPanel(config=cfg)
    panel.person_data_changed          # Signal()
    panel.refresh()                    # reload folder list
    panel.retranslate()                # after language change
    panel._reload_current_face_data()  # after external assignment change
"""

from __future__ import annotations

from app.ui.panels.image_browser import ImageBrowserPanel
from app.ui.panels.image_browser.helpers import (  # noqa: F401
    _FaceData,
    _BboxEdit,
    _TREE_THUMB_SIZE,
    _ROLE_TYPE,
    _ROLE_PAYLOAD,
    _ROLE_CACHE,
    _ROLE_FILE_PATH,
    _ROLE_TREE_NODE,
    _DRIVE_PREFETCH_AHEAD,
    _UF_HIDDEN,
    _is_visible_dir,
    _read_exif_date,
    _RE_FILENAME_DATETIME,
    _RE_FILENAME_DATE,
    _parse_date_from_filename,
    _bgr_to_qpixmap,
    _get_pil_font,
    _PENDING_RGB,
    _draw_faces,
    _hline,
    _overlay_group,
    _TreeNode,
    _build_folder_tree,
)
from app.ui.panels.image_browser.widgets import (  # noqa: F401
    _FocusLineEdit,
    _PendingActionBar,
    _InlineFaceEditor,
    _BrowserTree,
    _DrawableImageLabel,
)

__all__ = ["ImageBrowserPanel"]
