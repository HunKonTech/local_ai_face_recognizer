"""XMP editing for Face-Local's owned properties (stdlib ElementTree).

Face-Local owns three XMP properties on files it exports:

* ``facelocal:data`` — the ``facelocal.faces.v1`` JSON document,
* ``MP:RegionInfo`` — Microsoft Photo face regions (Windows "People"),
* ``mwg-rs:Regions`` — MWG face regions (Lightroom, digiKam, …).

:func:`edit_xmp` removes those properties wherever they are and writes the new
values into one ``rdf:Description``; every other property, description and
namespace prefix of the existing packet is kept.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

NS_X = "adobe:ns:meta/"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_FACELOCAL = "http://facelocal.app/ns/faces/1.0/"
NS_MP = "http://ns.microsoft.com/photo/1.2/"
NS_MPRI = "http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
NS_MPREG = "http://ns.microsoft.com/photo/1.2/t/Region#"
NS_MWGRS = "http://www.metadataworkinggroup.com/schemas/regions/"
NS_STDIM = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"
NS_STAREA = "http://ns.adobe.com/xmp/sType/Area#"

_PREFIXES = {
    "x": NS_X,
    "rdf": NS_RDF,
    "facelocal": NS_FACELOCAL,
    "MP": NS_MP,
    "MPRI": NS_MPRI,
    "MPReg": NS_MPREG,
    "mwg-rs": NS_MWGRS,
    "stDim": NS_STDIM,
    "stArea": NS_STAREA,
}

_XPACKET_ID = "W5M0MpCehiHzreSzNTczkc9d"

_OWNED = {
    f"{{{NS_FACELOCAL}}}data",
    f"{{{NS_MP}}}RegionInfo",
    f"{{{NS_MWGRS}}}Regions",
}

_RDF = f"{{{NS_RDF}}}"


class XmpParseError(ValueError):
    """The existing XMP packet cannot be parsed safely."""


@dataclass
class FaceRegion:
    """A named face rectangle, normalized (0..1) to the oriented image."""

    name: str
    x: float  # left
    y: float  # top
    w: float
    h: float


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def oriented_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    """Size of the image as displayed after applying EXIF *orientation*."""
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def regions_from_payload(payload: dict, oriented_w: int, oriented_h: int) -> list[FaceRegion]:
    """Named regions for the known persons of a ``facelocal.faces.v1`` payload.

    Boxes in the payload are already in the oriented (upright) pixel space.
    Faces without a name (unknown, or names excluded) or without a box get no
    region.
    """
    if oriented_w <= 0 or oriented_h <= 0:
        return []
    out: list[FaceRegion] = []
    for face in payload.get("faces") or []:
        name = face.get("person_name")
        box = face.get("box")
        if face.get("unknown") or not name or not box:
            continue

        def clamp(v: float) -> float:
            return round(min(max(v, 0.0), 1.0), 6)

        x0 = clamp(box["x"] / oriented_w)
        y0 = clamp(box["y"] / oriented_h)
        x1 = clamp((box["x"] + box["width"]) / oriented_w)
        y1 = clamp((box["y"] + box["height"]) / oriented_h)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(FaceRegion(str(name), x0, y0, round(x1 - x0, 6), round(y1 - y0, 6)))
    return out


# ---------------------------------------------------------------------------
# Parsing / serialisation
# ---------------------------------------------------------------------------

def _register_prefixes(prefixes: dict[str, str]) -> None:
    for prefix, uri in prefixes.items():
        try:
            ET.register_namespace(prefix, uri)
        except ValueError:
            pass  # reserved prefix such as "ns0" — ElementTree picks one


def _strip_packet(text: str) -> str:
    text = text.lstrip("﻿")
    text = re.sub(r"<\?xpacket[^>]*\?>", "", text)
    return text.strip().rstrip("\x00 \n\r\t")


def _parse(existing) -> tuple[ET.Element, ET.Element]:
    """Return ``(root, rdf_RDF)`` of an existing packet."""
    if isinstance(existing, (bytes, bytearray)):
        try:
            text = bytes(existing).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise XmpParseError("existing XMP is not valid UTF-8") from exc
    else:
        text = str(existing)
    body = _strip_packet(text)
    if not body:
        raise XmpParseError("existing XMP is empty")

    import io

    prefixes: dict[str, str] = {}
    try:
        for _event, (prefix, uri) in ET.iterparse(io.StringIO(body), events=("start-ns",)):
            if prefix and prefix not in prefixes:
                prefixes[prefix] = uri
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise XmpParseError(f"existing XMP unreadable: {exc}") from exc
    _register_prefixes(prefixes)

    if root.tag == f"{_RDF}RDF":
        return root, root
    rdf = root.find(f"{_RDF}RDF")
    if rdf is None:
        raise XmpParseError("existing XMP has no rdf:RDF")
    return root, rdf


def _new_root() -> tuple[ET.Element, ET.Element]:
    root = ET.Element(f"{{{NS_X}}}xmpmeta")
    rdf = ET.SubElement(root, f"{_RDF}RDF")
    return root, rdf


def _serialise(root: ET.Element) -> bytes:
    body = ET.tostring(root, encoding="unicode")
    return (
        f'<?xpacket begin="﻿" id="{_XPACKET_ID}"?>'
        f"{body}"
        '<?xpacket end="w"?>'
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Building owned properties
# ---------------------------------------------------------------------------

def _mp_region_info(regions: list[FaceRegion]) -> ET.Element:
    info = ET.Element(f"{{{NS_MP}}}RegionInfo", {f"{_RDF}parseType": "Resource"})
    regs = ET.SubElement(info, f"{{{NS_MPRI}}}Regions")
    bag = ET.SubElement(regs, f"{_RDF}Bag")
    for r in regions:
        li = ET.SubElement(bag, f"{_RDF}li", {f"{_RDF}parseType": "Resource"})
        ET.SubElement(li, f"{{{NS_MPREG}}}Rectangle").text = (
            f"{r.x:.6f}, {r.y:.6f}, {r.w:.6f}, {r.h:.6f}"
        )
        ET.SubElement(li, f"{{{NS_MPREG}}}PersonDisplayName").text = r.name
    return info


def _mwg_regions(regions: list[FaceRegion], width: int, height: int) -> ET.Element:
    top = ET.Element(f"{{{NS_MWGRS}}}Regions", {f"{_RDF}parseType": "Resource"})
    ET.SubElement(
        top,
        f"{{{NS_MWGRS}}}AppliedToDimensions",
        {
            f"{{{NS_STDIM}}}w": str(int(width)),
            f"{{{NS_STDIM}}}h": str(int(height)),
            f"{{{NS_STDIM}}}unit": "pixel",
        },
    )
    region_list = ET.SubElement(top, f"{{{NS_MWGRS}}}RegionList")
    bag = ET.SubElement(region_list, f"{_RDF}Bag")
    for r in regions:
        li = ET.SubElement(bag, f"{_RDF}li", {f"{_RDF}parseType": "Resource"})
        ET.SubElement(li, f"{{{NS_MWGRS}}}Name").text = r.name
        ET.SubElement(li, f"{{{NS_MWGRS}}}Type").text = "Face"
        ET.SubElement(
            li,
            f"{{{NS_MWGRS}}}Area",
            {
                # MWG areas are centre-based.
                f"{{{NS_STAREA}}}x": f"{r.x + r.w / 2:.6f}",
                f"{{{NS_STAREA}}}y": f"{r.y + r.h / 2:.6f}",
                f"{{{NS_STAREA}}}w": f"{r.w:.6f}",
                f"{{{NS_STAREA}}}h": f"{r.h:.6f}",
                f"{{{NS_STAREA}}}unit": "normalized",
            },
        )
    return top


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def edit_xmp(
    existing,
    *,
    json_text: Optional[str],
    regions: list[FaceRegion],
    applied_to: tuple[int, int],
) -> bytes:
    """Return a new XMP packet with Face-Local's owned properties replaced.

    Args:
        existing: The current packet (bytes/str) or None/empty for a new one.
        json_text: The ``facelocal.faces.v1`` JSON, or None to remove it.
        regions: Named face regions; an empty list removes both region sets.
        applied_to: Oriented ``(width, height)`` the regions refer to.

    Raises:
        XmpParseError: *existing* is present but cannot be parsed; the caller
            must then leave the file untouched rather than drop foreign XMP.
    """
    _register_prefixes(_PREFIXES)
    if existing:
        root, rdf = _parse(existing)
    else:
        root, rdf = _new_root()

    # Drop owned properties (child elements and attribute shorthand).
    for desc in list(rdf.findall(f"{_RDF}Description")):
        for child in list(desc):
            if child.tag in _OWNED:
                desc.remove(child)
        for attr in list(desc.attrib):
            if attr in _OWNED:
                del desc.attrib[attr]
        own_attrs = {a for a in desc.attrib if a != f"{_RDF}about"}
        if len(desc) == 0 and not own_attrs and desc.text is None:
            rdf.remove(desc)

    if json_text is not None or regions:
        _register_prefixes(_PREFIXES)
        desc = ET.SubElement(rdf, f"{_RDF}Description", {f"{_RDF}about": ""})
        if json_text is not None:
            ET.SubElement(desc, f"{{{NS_FACELOCAL}}}data").text = json_text
        if regions:
            desc.append(_mp_region_info(regions))
            desc.append(_mwg_regions(regions, *applied_to))

    return _serialise(root)


def read_facelocal_json(xmp) -> Optional[str]:
    """The ``facelocal:data`` text of an XMP packet, or None."""
    if not xmp:
        return None
    try:
        _root, rdf = _parse(xmp)
    except XmpParseError:
        return _read_facelocal_regex(xmp)
    for desc in rdf.findall(f"{_RDF}Description"):
        attr = desc.attrib.get(f"{{{NS_FACELOCAL}}}data")
        if attr:
            return attr
        node = desc.find(f"{{{NS_FACELOCAL}}}data")
        if node is not None and node.text:
            return node.text
    return None


_LEGACY_RE = re.compile(r"<facelocal:data>(.*?)</facelocal:data>", re.DOTALL)


def _read_facelocal_regex(xmp) -> Optional[str]:
    text = xmp.decode("utf-8", errors="replace") if isinstance(xmp, (bytes, bytearray)) else str(xmp)
    m = _LEGACY_RE.search(text)
    if not m:
        return None
    return m.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def read_region_names(xmp) -> tuple[list[str], list[str]]:
    """Person names in ``(MP regions, MWG regions)`` — used by tests/diagnostics."""
    if not xmp:
        return [], []
    _root, rdf = _parse(xmp)
    mp = [e.text or "" for e in rdf.iter(f"{{{NS_MPREG}}}PersonDisplayName")]
    mwg = [e.text or "" for e in rdf.iter(f"{{{NS_MWGRS}}}Name")]
    return mp, mwg
