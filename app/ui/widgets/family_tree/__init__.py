"""Graphical family-tree view package (mixins split by topic).

``view`` holds :class:`FamilyTreeView`; ``style`` colours, ``geometry`` Qt-free
helpers, ``connectors`` lines, ``node_box`` person boxes, ``photos`` lazy photo
loading and ``interaction`` input handling.
"""

from app.ui.widgets.family_tree.view import FamilyTreeView

__all__ = ["FamilyTreeView"]
