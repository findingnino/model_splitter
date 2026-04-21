#!/usr/bin/env python
"""
split_model.py  —  Scale a 3D model (STL/3MF) to a target real-world bounding
box and slice it into grid-aligned pieces that fit a given printer build
volume, with wooden-dowel holes drilled through every shared cut face so the
pieces can be reassembled with standard Home Depot / Lowes dowels.

Outputs to ./split_output/ (or wherever the user chooses):
    piece_ix_iy_iz.stl   one STL per non-empty grid cell
    assembly_map.png     three orthographic views with piece labels
    README.txt           piece list, dowel count, slicer recommendations

Run interactively:
    python split_model.py

Or as a one-liner (skips the prompts):
    python split_model.py INPUT.stl --target 2000 720 600 --print-volume 250 250 250
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

# Force UTF-8 console output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import shapely.geometry as sg
from shapely.ops import unary_union
import trimesh
from PIL import Image, ImageDraw, ImageFont


# ---------- constants ----------

# Standard wooden dowel diameters stocked at Home Depot / Lowes
STANDARD_DOWELS: list[tuple[str, float]] = [
    ("1/8\"",  3.175),
    ("3/16\"", 4.7625),
    ("1/4\"",  6.35),
    ("5/16\"", 7.9375),
    ("3/8\"",  9.525),
    ("1/2\"",  12.7),
    ("5/8\"",  15.875),
    ("3/4\"",  19.05),
    ("1\"",    25.4),
]

HOLE_CLEARANCE_MM       = 0.4    # hole dia = dowel dia + this (slip fit)
DOWEL_DEPTH_FACTOR      = 2.0    # hole depth into each piece = this * dowel dia
DOWEL_WALL_MARGIN_MM    = 1.5    # min wall thickness between dowel and cut edge
DOWEL_SPACING_FACTOR    = 3.0    # minimum center-to-center distance = this * dia
MIN_DOWELS_PER_FACE     = 2      # target at least this many per shared face for alignment
MAX_DOWELS_PER_FACE     = 6      # cap, even on huge faces

# Default set of sizes the program may choose from when user picks "auto"
DEFAULT_DOWEL_SET_MM = [6.35, 9.525, 12.7, 15.875]  # 1/4", 3/8", 1/2", 5/8"

# Wall markings: unique letter label engraved into each shared cut face
MARK_ENGRAVE_DEPTH_MM   = 0.6    # how deep the letter is pressed into each piece
MARK_TEXT_HEIGHT_MIN_MM = 4.0    # too small to read below this
MARK_TEXT_HEIGHT_MAX_MM = 18.0   # cap so it doesn't dominate the face
MARK_FONT_FAMILY        = "DejaVu Sans"  # bundled with matplotlib
PRINT_SAFETY_FRACTION   = 0.10   # 10 % less than the printer's stated volume
DEBRIS_VERT_RATIO       = 1e-3   # drop conn-components smaller than this
MIN_PIECE_VOLUME_MM3    = 20.0   # drop empty/tiny leftovers after cutting
VENT_HOLE_DIAM_MM       = 4.0    # for hollow mode


# ---------- user I/O ----------

def _to_mm(token: str) -> float:
    """Accept '250', '250mm', '10in', '10\"', '25cm'."""
    t = token.strip().lower().replace(" ", "")
    if t.endswith("mm"):
        return float(t[:-2])
    if t.endswith("cm"):
        return float(t[:-2]) * 10.0
    if t.endswith("in") or t.endswith('"'):
        return float(t[:-2] if t.endswith("in") else t[:-1]) * 25.4
    return float(t)


def _parse_triplet(raw: str) -> tuple[float, float, float]:
    parts = [p for p in raw.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError(f"need 3 numbers, got {len(parts)}")
    vals = [_to_mm(p) for p in parts]
    if any(v <= 0 for v in vals):
        raise ValueError("dimensions must be positive")
    return (vals[0], vals[1], vals[2])


def _prompt_triplet(label: str, default: tuple[float, float, float] | None) -> tuple[float, float, float]:
    default_str = ""
    if default is not None:
        default_str = f" [{default[0]:g} {default[1]:g} {default[2]:g}]"
    while True:
        raw = input(f"{label}{default_str}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return _parse_triplet(raw)
        except ValueError as e:
            print(f"  couldn't parse: {e}")


def _prompt_yn(label: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    raw = input(f"{label} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _default_dowel_set(cell_size: tuple[float, float, float]) -> list[tuple[str, float]]:
    """Default auto-selection set, scaled to the cell size.

    For small cells we exclude sizes that would never fit; for very large cells
    we add larger sizes. Always preserves a reasonable spread so the per-face
    selector has options.
    """
    smallest = min(cell_size)
    # Rough upper bound: dowel diam should fit within ~1/3 of smallest cell dim
    upper = smallest / 3.0
    candidates = [d for d in DEFAULT_DOWEL_SET_MM if d <= max(upper, 6.35)]
    if not candidates:
        candidates = [DEFAULT_DOWEL_SET_MM[0]]
    # Convert to (label, mm) form, matching STANDARD_DOWELS labels where possible
    out: list[tuple[str, float]] = []
    for mm in candidates:
        match = next((p for p in STANDARD_DOWELS if abs(p[1] - mm) < 0.05), None)
        out.append(match if match else (f"{mm:.2f}mm", mm))
    return out


def _parse_dowel_list(raw: str) -> list[tuple[str, float]]:
    """Parse user input like '1/4,3/8,1/2' or '6.35, 9.525, 12.7' or '3,5,6'.

    Numeric tokens >= 10 are treated as index into STANDARD_DOWELS; smaller ones
    as raw mm. Fractional tokens like '1/2\"' are matched to STANDARD_DOWELS.
    """
    out: list[tuple[str, float]] = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip().rstrip('"')
        if not tok:
            continue
        # Fractional (looks like 'N/M')
        if "/" in tok:
            match = next((p for p in STANDARD_DOWELS
                          if p[0].rstrip('"') == tok), None)
            if match:
                out.append(match)
                continue
        try:
            v = float(tok)
            # Treat small integers (1-9) as indices into STANDARD_DOWELS
            if v == int(v) and 1 <= v <= len(STANDARD_DOWELS):
                out.append(STANDARD_DOWELS[int(v) - 1])
            else:
                match = next((p for p in STANDARD_DOWELS if abs(p[1] - v) < 0.05), None)
                out.append(match if match else (f"{v:.2f}mm", v))
        except ValueError:
            print(f"  couldn't parse dowel token '{tok}' - skipping")
    # Dedup by diameter, keep first label
    seen = set()
    deduped = []
    for lbl, mm in out:
        key = round(mm, 3)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((lbl, mm))
    return deduped


def _prompt_dowel_set(cell_size: tuple[float, float, float]) -> list[tuple[str, float]]:
    default_set = _default_dowel_set(cell_size)
    default_desc = ", ".join(lbl for lbl, _ in default_set)
    print("\nDowel sizing:")
    print("  The program can auto-pick per face from a SET of sizes.")
    print("  Give one size to force it everywhere, or several to let the")
    print("  program pick the largest size that still fits >=2 dowels per face.")
    print("\nStandard sizes:")
    for idx, (lbl, mm) in enumerate(STANDARD_DOWELS, start=1):
        tag = "  (in default auto set)" if any(abs(mm - d[1]) < 0.05 for d in default_set) else ""
        print(f"  {idx}. {lbl:6s}  ({mm:6.3f} mm){tag}")
    while True:
        raw = input(f"Enter comma-separated sizes (numbers, mm, or fractions) [{default_desc}]: ").strip()
        if not raw:
            return default_set
        parsed = _parse_dowel_list(raw)
        if parsed:
            return parsed
        print("  couldn't parse any valid sizes, try again")


# ---------- mesh helpers ----------

def load_and_clean(path: Path) -> trimesh.Trimesh:
    print(f"\nLoading {path} ...")
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"couldn't load single mesh from {path}")
    print(f"  {len(mesh.vertices):,} verts  /  {len(mesh.faces):,} faces")
    print(f"  watertight={mesh.is_watertight}  is_volume={mesh.is_volume}")

    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        biggest = max(len(p.vertices) for p in parts)
        thresh = biggest * DEBRIS_VERT_RATIO
        kept = [p for p in parts if len(p.vertices) >= thresh]
        dropped = len(parts) - len(kept)
        if dropped:
            print(f"  dropped {dropped} tiny debris component(s)")
        mesh = kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)

    if not mesh.is_watertight:
        print("  filling holes ...")
        mesh.fill_holes()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()

    if not mesh.is_volume:
        print("  WARNING: mesh is not a valid volume — booleans may fail.")
    return mesh


def scale_to_bbox(mesh: trimesh.Trimesh, target: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds[0])
    factors = np.array(target) / mesh.extents
    ratio = factors.max() / factors.min()
    print(f"  scale factors (x,y,z) = ({factors[0]:.3f}, {factors[1]:.3f}, {factors[2]:.3f})")
    if ratio > 1.10:
        print(f"  NOTE: non-uniform stretch {ratio:.2f}× — model will be distorted. "
              "Use an aspect-ratio-preserving target if you want undistorted geometry.")
    mesh.apply_scale(factors)
    return mesh


def _cell_box(origin: tuple[float, float, float], size: tuple[float, float, float]) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=size)
    b.apply_translation(np.asarray(origin) + np.asarray(size) / 2.0)
    return b


def cut_into_pieces(
    mesh: trimesh.Trimesh,
    divisions: tuple[int, int, int],
    cell_size: tuple[float, float, float],
) -> dict[tuple[int, int, int], trimesh.Trimesh]:
    pieces: dict[tuple[int, int, int], trimesh.Trimesh] = {}
    total = divisions[0] * divisions[1] * divisions[2]
    n = 0
    for ix in range(divisions[0]):
        for iy in range(divisions[1]):
            for iz in range(divisions[2]):
                n += 1
                origin = (ix * cell_size[0], iy * cell_size[1], iz * cell_size[2])
                box = _cell_box(origin, cell_size)
                try:
                    piece = mesh.intersection(box)
                except Exception as e:
                    print(f"  [{n:>3}/{total}] ({ix},{iy},{iz}) intersection failed: {e}")
                    continue
                if piece is None or len(piece.vertices) == 0:
                    continue
                vol = piece.volume if piece.is_volume else 0.0
                if vol < MIN_PIECE_VOLUME_MM3:
                    continue
                pieces[(ix, iy, iz)] = piece
                print(f"  [{n:>3}/{total}] ({ix},{iy},{iz}) "
                      f"{len(piece.vertices):>6,} v   vol={vol/1000:7.1f} cm³")
    return pieces


# ---------- dowel placement ----------

SECTION_EPSILON_MM = 0.05  # offset into the piece so we don't section exactly on a cut face


def _section_polygon(mesh: trimesh.Trimesh, axis: int, coord: float,
                     offset_sign: int = -1):
    """Return (shapely polygon in plane-2D, 4x4 transform 2D->3D) or None.

    `coord` is the world-space cut plane. We section slightly offset by
    `offset_sign * SECTION_EPSILON_MM` along the axis to avoid the degenerate
    case where the cut plane coincides with a flat mesh face.
    """
    effective_coord = coord + offset_sign * SECTION_EPSILON_MM
    origin = [0.0, 0.0, 0.0]; origin[axis] = effective_coord
    normal = [0.0, 0.0, 0.0]; normal[axis] = 1.0
    sec3d = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec3d is None:
        return None
    try:
        # Newer trimesh exposes `to_2D`; older uses `to_planar`. Prefer `to_2D`.
        if hasattr(sec3d, "to_2D"):
            sec2d, T = sec3d.to_2D()
        else:
            sec2d, T = sec3d.to_planar()
    except Exception:
        return None
    polys = [p for p in sec2d.polygons_full if p.is_valid and p.area > 0.1]
    if not polys:
        return None
    # Rewrite the transform so the returned 3D positions sit exactly on the
    # cut plane (not on effective_coord). We do this by zeroing out the axis
    # component of the transform's translation and reinjecting `coord`.
    T = T.copy()
    T[axis, 3] = coord
    return unary_union(polys), T


def _target_count_for_area(area_mm2: float) -> int:
    """Ideal dowel count based on the shared-face cross-section area."""
    if area_mm2 < 1_000:   return max(MIN_DOWELS_PER_FACE, 2)
    if area_mm2 < 5_000:   return 3
    if area_mm2 < 15_000:  return 4
    if area_mm2 < 35_000:  return 5
    return MAX_DOWELS_PER_FACE


def _try_place_dowels(poly, diam: float, target_n: int) -> list[tuple[float, float]]:
    """Greedy farthest-point placement of up to target_n dowels inside `poly`.

    Returns the list of (u, v) center coordinates. `poly` is the raw
    cross-section polygon; we handle the erosion by dowel_radius + wall
    internally so the caller doesn't have to.
    """
    r = diam / 2.0
    erode_by = r + DOWEL_WALL_MARGIN_MM
    min_spacing = diam * DOWEL_SPACING_FACTOR

    regions = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    regions.sort(key=lambda g: -g.area)

    picks: list[tuple[float, float]] = []

    for region in regions:
        if len(picks) >= target_n:
            break
        eroded = region.buffer(-erode_by)
        if eroded.is_empty:
            continue
        sub_regions = list(eroded.geoms) if eroded.geom_type == "MultiPolygon" else [eroded]
        sub_regions.sort(key=lambda g: -g.area)

        for sub in sub_regions:
            if len(picks) >= target_n:
                break
            if sub.is_empty:
                continue
            # Place first dowel at representative point (guaranteed inside)
            rep = sub.representative_point()
            picks.append((rep.x, rep.y))
            if len(picks) >= target_n:
                break

            # Farthest-point sampling for subsequent dowels in this sub-region
            minx, miny, maxx, maxy = sub.bounds
            step = max(min_spacing * 0.35, 0.5)
            while len(picks) < target_n:
                best_pt = None
                best_dist = 0.0
                y = miny + step / 2.0
                while y <= maxy:
                    x = minx + step / 2.0
                    while x <= maxx:
                        if sub.contains(sg.Point(x, y)):
                            d_near = min(math.hypot(x - px, y - py) for px, py in picks)
                            if d_near >= min_spacing and d_near > best_dist:
                                best_dist = d_near
                                best_pt = (x, y)
                        x += step
                    y += step
                if best_pt is None:
                    break
                picks.append(best_pt)
    return picks[:target_n]


def _select_dowel_for_face(poly, allowed: list[tuple[str, float]]
                           ) -> tuple[str, float, list[tuple[float, float]]]:
    """Pick the best dowel size + positions for a single shared cut face.

    Strategy:
      1. From largest to smallest, find the first size that can place >=
         MIN_DOWELS_PER_FACE dowels. Place up to the area-based target.
      2. If no size can place >= MIN_DOWELS_PER_FACE, fall back to the
         largest size that fits 1 dowel (still better than nothing).
      3. If nothing fits, return empty.
    """
    # Filter invalid polygons
    if poly.is_empty:
        return ("", 0.0, [])

    area = poly.area
    sizes_desc = sorted(allowed, key=lambda s: -s[1])

    # Phase 1: largest size that fits >= MIN_DOWELS_PER_FACE
    for label, diam in sizes_desc:
        target_n = _target_count_for_area(area)
        positions = _try_place_dowels(poly, diam, target_n)
        if len(positions) >= MIN_DOWELS_PER_FACE:
            return (label, diam, positions)

    # Phase 2: largest size that fits 1
    for label, diam in sizes_desc:
        positions = _try_place_dowels(poly, diam, 1)
        if positions:
            return (label, diam, positions)

    return ("", 0.0, [])


def _make_cylinder_world(center_xyz: np.ndarray, axis: int,
                         radius: float, length: float) -> trimesh.Trimesh:
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
    if axis == 0:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 1:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation(center_xyz)
    return cyl


# ---------- wall-marking labels ----------

def _label_for_index(i: int) -> str:
    """0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, 27 -> AB, ..."""
    result = ""
    n = i + 1
    while n > 0:
        n -= 1
        result = chr(ord("A") + n % 26) + result
        n //= 26
    return result


def _text_to_polygon(text: str, size_mm: float):
    """Convert a text string to a shapely (Multi)Polygon."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    fp = FontProperties(family=MARK_FONT_FAMILY)
    tp = TextPath((0, 0), text, size=size_mm, prop=fp)
    raw_polys = tp.to_polygons()
    if not raw_polys:
        return None

    polys = []
    for p in raw_polys:
        if len(p) < 3:
            continue
        sp = sg.Polygon(p).buffer(0)
        if sp.is_empty:
            continue
        if sp.geom_type == "Polygon":
            polys.append(sp)
        elif sp.geom_type == "MultiPolygon":
            polys.extend(sp.geoms)
    if not polys:
        return None

    # Letters like A, B, D, O, P, Q, R have inner holes. Use polygon-in-polygon
    # `within` rather than rep-point containment, because the representative
    # point of the outer glyph can fall inside the hole geometrically.
    depths = []
    for i, p in enumerate(polys):
        depth = sum(1 for j, q in enumerate(polys) if i != j and p.within(q))
        depths.append(depth)

    outers = [p for p, d in zip(polys, depths) if d % 2 == 0]
    holes = [p for p, d in zip(polys, depths) if d % 2 == 1]

    if not outers:
        return None

    outer_union = unary_union(outers)
    if holes:
        outer_union = outer_union.difference(unary_union(holes))
    return outer_union


def _text_mesh_for_face(label: str, face_poly, dowel_positions: list[tuple[float, float]],
                        largest_dowel_diam: float, engrave_depth: float
                        ) -> tuple[trimesh.Trimesh | None, tuple[float, float] | None, float]:
    """Build a 3D mesh of extruded text for engraving into a cut face.

    Returns (mesh, position_2d, chosen_size_mm) or (None, None, 0.0) if nothing
    readable fits.

    Strategy: try several (exclusion_radius, fill_ratio) combinations in order
    from strict to permissive. A permissive pass may place text overlapping a
    dowel hole, which is acceptable — the hole takes priority and you still
    see most of the letter around it.
    """
    if face_poly.is_empty:
        return None, None, 0.0

    from shapely.affinity import translate as shp_translate

    # (exclusion radius multiplier on dowel diam, text fill ratio within region)
    attempts = [
        (1.2, 0.7),   # strict: big exclusion, text fills 70% of region
        (0.9, 0.6),   # looser exclusion
        (0.0, 0.55),  # no exclusion (letter may overlap holes)
        (0.0, 0.40),  # final fallback: smaller text, no exclusion
    ]

    for excl_mult, fill_ratio in attempts:
        exclusion_r = largest_dowel_diam * excl_mult if excl_mult > 0 else 0.0
        free = face_poly
        if exclusion_r > 0:
            for (u, v) in dowel_positions:
                try:
                    free = free.difference(
                        sg.Point(u, v).buffer(exclusion_r, resolution=16)
                    )
                except Exception:
                    pass
            if free.is_empty:
                continue

        regions = [free] if free.geom_type == "Polygon" else list(free.geoms)
        regions = [r for r in regions if not r.is_empty]
        if not regions:
            continue
        target = max(regions, key=lambda r: r.area)

        minx, miny, maxx, maxy = target.bounds
        w, h = maxx - minx, maxy - miny
        char_width_ratio = 0.6
        fit_by_h = h * fill_ratio
        fit_by_w = (w * fill_ratio) / (char_width_ratio * max(1, len(label)))
        size = min(fit_by_h, fit_by_w, MARK_TEXT_HEIGHT_MAX_MM)
        if size < MARK_TEXT_HEIGHT_MIN_MM:
            continue

        rep = target.representative_point()
        cx, cy = rep.x, rep.y

        poly_2d = _text_to_polygon(label, size)
        if poly_2d is None or poly_2d.is_empty:
            continue

        tminx, tminy, tmaxx, tmaxy = poly_2d.bounds
        dx = cx - (tminx + tmaxx) / 2.0
        dy = cy - (tminy + tmaxy) / 2.0
        poly_2d = shp_translate(poly_2d, xoff=dx, yoff=dy)

        # Skip if the text would mostly lie outside the face material —
        # otherwise it's a wasted op (parts of the letter would just hang in
        # air). Don't actually clip; clipped shapes can produce degenerate
        # extrusions that break later booleans.
        try:
            inside = poly_2d.intersection(face_poly).area
        except Exception:
            inside = poly_2d.area
        if inside < poly_2d.area * 0.5:
            continue

        try:
            mesh = trimesh.creation.extrude_polygon(poly_2d, height=2 * engrave_depth)
        except Exception:
            continue
        mesh.apply_translation([0.0, 0.0, -engrave_depth])
        return mesh, (cx, cy), size

    return None, None, 0.0


def drill_dowel_holes(
    pieces: dict[tuple[int, int, int], trimesh.Trimesh],
    divisions: tuple[int, int, int],
    cell_size: tuple[float, float, float],
    allowed_sizes: list[tuple[str, float]],
    add_markings: bool = True,
) -> tuple[dict[tuple[int, int, int], trimesh.Trimesh], list[dict]]:
    """
    For every interior grid plane, pick the best dowel size per shared face
    from `allowed_sizes`, place the dowel pattern, and (optionally) engrave
    a unique letter label into each face. All cuts (cylinders + text meshes)
    are subtracted from the adjacent pieces in one boolean per piece at the end.

    Returns (updated pieces dict, list of face records). Each face record:
      {axis, plane_coord, cells, size_label, diam_mm, count, area_mm2, mark}
    """
    per_piece: dict[tuple[int, int, int], list[trimesh.Trimesh]] = defaultdict(list)
    face_records: list[dict] = []

    # Assign labels in a deterministic axis-major order
    face_index = 0

    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for i in range(1, divisions[axis]):
            plane_coord = i * cell_size[axis]
            for ia in range(divisions[other[0]]):
                for ib in range(divisions[other[1]]):
                    key_lo = [0, 0, 0]
                    key_lo[axis] = i - 1
                    key_lo[other[0]] = ia
                    key_lo[other[1]] = ib
                    key_hi = list(key_lo); key_hi[axis] = i
                    key_lo, key_hi = tuple(key_lo), tuple(key_hi)

                    if key_lo not in pieces or key_hi not in pieces:
                        continue

                    res = _section_polygon(pieces[key_lo], axis, plane_coord)
                    if res is None:
                        continue
                    poly, T = res

                    size_label, diam, pts_2d = _select_dowel_for_face(poly, allowed_sizes)

                    # Allocate the next label no matter what — we'll still try to
                    # engrave even if the face is too small for a dowel.
                    mark = _label_for_index(face_index)
                    face_index += 1

                    # Dowel cylinders
                    if pts_2d:
                        hole_r = (diam + HOLE_CLEARANCE_MM) / 2.0
                        cyl_len = diam * DOWEL_DEPTH_FACTOR * 2.0
                        for (u, v) in pts_2d:
                            pt_world = (T @ np.array([u, v, 0.0, 1.0]))[:3]
                            cyl = _make_cylinder_world(pt_world, axis, hole_r, cyl_len)
                            per_piece[key_lo].append(cyl)
                            per_piece[key_hi].append(cyl)

                    # Text engraving (subtracted on both sides so both pieces get
                    # a mirrored version of the same letter — matching halves).
                    mark_engraved = False
                    if add_markings:
                        text_mesh, _pos, text_size = _text_mesh_for_face(
                            mark, poly, pts_2d, diam if diam else 10.0,
                            MARK_ENGRAVE_DEPTH_MM,
                        )
                        if text_mesh is not None:
                            text_mesh.apply_transform(T)
                            per_piece[key_lo].append(text_mesh)
                            per_piece[key_hi].append(text_mesh)
                            mark_engraved = True

                    face_records.append({
                        "axis": axis, "plane_coord": plane_coord,
                        "cells": (key_lo, key_hi),
                        "size_label": size_label if pts_2d else "-",
                        "diam_mm": diam if pts_2d else 0.0,
                        "count": len(pts_2d),
                        "area_mm2": poly.area,
                        "mark": mark if mark_engraved else "-",
                    })

    n_total = sum(r["count"] for r in face_records)
    n_drilled = sum(1 for r in face_records if r["count"] > 0)
    n_empty = sum(1 for r in face_records if r["count"] == 0)
    n_marks = sum(1 for r in face_records if r["mark"] != "-")
    print(f"  {n_drilled} shared face(s) drilled, {n_total} dowel hole(s) total"
          + (f"  ({n_empty} face(s) too small for any dowel)" if n_empty else ""))
    if add_markings:
        print(f"  engraved {n_marks}/{len(face_records)} face labels")

    # Per-size summary
    by_size: dict[tuple[str, float], int] = defaultdict(int)
    for r in face_records:
        if r["count"]:
            by_size[(r["size_label"], r["diam_mm"])] += r["count"]
    for (lbl, diam), n in sorted(by_size.items(), key=lambda kv: kv[0][1]):
        print(f"    {lbl:8s} ({diam:6.3f} mm): {n} dowel(s)")

    out: dict[tuple[int, int, int], trimesh.Trimesh] = {}
    for key, piece in pieces.items():
        subs = per_piece.get(key, [])
        if not subs:
            out[key] = piece
            continue
        try:
            res = trimesh.boolean.difference([piece, *subs])
            if res is None or len(res.vertices) == 0:
                print(f"  WARNING: boolean returned empty for {key} - keeping original")
                out[key] = piece
            else:
                # Clean up boolean output so slicers don't choke on hairline
                # non-manifold edges left by the extrusion stamps.
                res.merge_vertices()
                res.remove_unreferenced_vertices()
                if not res.is_watertight:
                    res.fill_holes()
                out[key] = res
        except Exception as e:
            print(f"  WARNING: boolean failed for {key}: {e} - keeping original")
            out[key] = piece
    return out, face_records


# ---------- hollow-mode vent holes ----------

def add_vent_holes(
    pieces: dict[tuple[int, int, int], trimesh.Trimesh],
    diam: float = VENT_HOLE_DIAM_MM,
) -> dict[tuple[int, int, int], trimesh.Trimesh]:
    r = diam / 2.0
    out = {}
    for key, piece in pieces.items():
        cx, cy = piece.centroid[0], piece.centroid[1]
        z0, z1 = piece.bounds[0, 2] - 1.0, piece.bounds[1, 2] + 1.0
        cyl = trimesh.creation.cylinder(radius=r, height=(z1 - z0), sections=16)
        cyl.apply_translation([cx, cy, (z0 + z1) / 2.0])
        try:
            res = trimesh.boolean.difference([piece, cyl])
            out[key] = res if res is not None and len(res.vertices) else piece
        except Exception:
            out[key] = piece
    return out


# ---------- export + docs ----------

def export_pieces(
    pieces: dict[tuple[int, int, int], trimesh.Trimesh],
    out_dir: Path,
) -> list[tuple[tuple[int, int, int], Path, float]]:
    records = []
    for key in sorted(pieces):
        piece = pieces[key]
        fname = out_dir / f"piece_x{key[0]}_y{key[1]}_z{key[2]}.stl"
        piece.export(str(fname))
        vol_cm3 = (piece.volume / 1000.0) if piece.is_volume else 0.0
        records.append((key, fname, vol_cm3))
    return records


def write_assembly_map(
    records: list[tuple[tuple[int, int, int], Path, float]],
    divisions: tuple[int, int, int],
    cell_size: tuple[float, float, float],
    out_path: Path,
) -> None:
    occupied = {rec[0] for rec in records}
    axis_name = ("X", "Y", "Z")
    cell_px = 72
    pad = 24
    title_h = 40
    gap = 40

    def view(fixed_axis: int) -> Image.Image:
        ax = [a for a in range(3) if a != fixed_axis]
        labels = ("X", "Y", "Z")
        cols = divisions[ax[0]]
        rows = divisions[ax[1]]
        title = (f"Looking along {labels[fixed_axis]}  "
                 f"(horiz = {labels[ax[0]]}, vert = {labels[ax[1]]})")

        try:
            title_font = ImageFont.truetype("arial.ttf", 14)
            cell_font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            title_font = cell_font = small_font = ImageFont.load_default()

        # Width: enough for both the grid and the title
        try:
            text_w = int(title_font.getlength(title)) + pad * 2
        except Exception:
            text_w = 400
        w = max(cols * cell_px + pad * 2, text_w)
        h = rows * cell_px + pad + title_h
        img = Image.new("RGB", (w, h), (248, 248, 248))
        d = ImageDraw.Draw(img)
        d.text((pad, 10), title, fill=(10, 10, 10), font=title_font)

        # count how many pieces project into each 2D cell
        projected: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for key in occupied:
            projected[(key[ax[0]], key[ax[1]])].append(key)

        grid_left = pad
        grid_top = title_h
        for a in range(cols):
            for b in range(rows):
                x0 = grid_left + a * cell_px
                y0 = grid_top + (rows - 1 - b) * cell_px   # flip so b=0 is bottom
                x1, y1 = x0 + cell_px, y0 + cell_px
                cell_keys = projected.get((a, b), [])
                fill = (220, 235, 255) if cell_keys else (238, 238, 238)
                d.rectangle([x0, y0, x1, y1], fill=fill, outline=(120, 120, 120))
                label = f"{labels[ax[0]]}={a}\n{labels[ax[1]]}={b}"
                d.text((x0 + 6, y0 + 4), label, fill=(30, 30, 30), font=cell_font)
                if cell_keys:
                    d.text((x0 + 6, y1 - 16), f"n={len(cell_keys)}",
                           fill=(60, 60, 120), font=small_font)
        return img

    imgs = [view(2), view(1), view(0)]
    total_w = sum(i.width for i in imgs) + gap * (len(imgs) - 1) + pad * 2
    total_h = max(i.height for i in imgs) + pad * 2
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    x = pad
    for img in imgs:
        canvas.paste(img, (x, pad))
        x += img.width + gap
    canvas.save(out_path)


def write_readme(
    records: list[tuple[tuple[int, int, int], Path, float]],
    divisions: tuple[int, int, int],
    cell_size: tuple[float, float, float],
    target: tuple[float, float, float],
    hollow: bool,
    allowed_sizes: list[tuple[str, float]],
    face_records: list[dict],
    out_path: Path,
) -> None:
    total_vol = sum(r[2] for r in records)
    allowed_desc = ", ".join(f"{lbl} ({mm:.3f} mm)" for lbl, mm in allowed_sizes)

    # Tally dowels by size
    by_size: dict[tuple[str, float], int] = defaultdict(int)
    faces_drilled = 0
    faces_too_small = 0
    for fr in face_records:
        if fr["count"] > 0:
            by_size[(fr["size_label"], fr["diam_mm"])] += fr["count"]
            faces_drilled += 1
        else:
            faces_too_small += 1
    total_dowels = sum(by_size.values())

    txt = f"""\
Model Splitter - build instructions
===================================

Target size (mm):    {target[0]:.1f} x {target[1]:.1f} x {target[2]:.1f}
Grid divisions:      {divisions[0]} x {divisions[1]} x {divisions[2]}
Cell size (mm):      {cell_size[0]:.1f} x {cell_size[1]:.1f} x {cell_size[2]:.1f}
Pieces produced:     {len(records)}
Total volume:        {total_vol:,.1f} cm^3  (~{total_vol*1.24/1000:.2f} kg at PLA 1.24 g/cc)
Hollow mode:         {"yes - each piece has a 4 mm vent hole" if hollow else "no"}

Shared cut faces:    {len(face_records)}
Faces drilled:       {faces_drilled}
Faces too small:     {faces_too_small}
Total dowels needed: {total_dowels}

Dowel sizes allowed: {allowed_desc}

Dowel shopping list
-------------------
"""
    if not by_size:
        txt += "  (no dowels placed)\n"
    else:
        for (lbl, diam), n in sorted(by_size.items(), key=lambda kv: kv[0][1]):
            cut_len = diam * DOWEL_DEPTH_FACTOR * 2 + 1.0
            txt += (f"  {lbl:8s} ({diam:6.3f} mm dia): "
                    f"cut {n} pieces at {cut_len:.0f} mm each "
                    f"(~{n * cut_len / 25.4:.1f} in total length required)\n")

    txt += "\nSlicer recommendations\n----------------------\n"
    if hollow:
        txt += textwrap.dedent("""\
          Infill:        0 %
          Walls:         4 perimeters minimum (5+ for structural pieces)
          Top layers:    0 (piece is sealed by its shell)
          Bottom layers: 3
          Supports:      as needed - orient pieces cut-face-down when possible
          The 4 mm vent hole lets trapped air escape during print. Don't plug it.
        """)
    else:
        txt += textwrap.dedent("""\
          Infill:        15 - 25 % gyroid or cubic (rigid, light)
          Walls:         3 perimeters
          Supports:      as needed - orient pieces cut-face-down when possible
        """)

    txt += "\nAssembly\n--------\n"
    txt += "1. Print every STL in this folder.\n"
    txt += "2. Cut dowels to the lengths listed in the shopping list above.\n"
    txt += "3. Glue dowels into one side's holes first (wood glue or CA); let set.\n"
    txt += "4. Dry-fit all pieces, then glue mating sides together.\n"
    txt += "5. See assembly_map.png for grid layout (three orthographic views).\n\n"

    txt += "Piece list\n----------\n"
    for key, path, vol in records:
        txt += f"  {path.name:40s}  cell ({key[0]},{key[1]},{key[2]})  {vol:7.1f} cm^3\n"

    txt += "\nPer-face breakdown\n------------------\n"
    txt += "  mark  axis  plane_mm    cells                       size    dowels   face_area_cm2\n"
    axis_name = ("X", "Y", "Z")
    for fr in face_records:
        lo, hi = fr["cells"]
        txt += (f"  {fr['mark']:>4s}  {axis_name[fr['axis']]}     {fr['plane_coord']:7.1f}  "
                f"{str(lo):<11s} <-> {str(hi):<11s} "
                f"{fr['size_label']:>7s}  {fr['count']:>5d}    {fr['area_mm2']/100.0:7.1f}\n")

    txt += "\nAssembly tip: matching faces share the same letter. Find the two\n"
    txt += "pieces whose cut faces both show 'A', mate them; then 'B', and so on.\n"

    out_path.write_text(txt, encoding="utf-8")


# ---------- glue ----------

def run(
    input_path: Path,
    target: tuple[float, float, float],
    print_volume: tuple[float, float, float],
    hollow: bool,
    allowed_sizes: list[tuple[str, float]],
    output_dir: Path,
    add_markings: bool = True,
) -> None:

    safe_volume = tuple(v * (1.0 - PRINT_SAFETY_FRACTION) for v in print_volume)
    divisions = tuple(max(1, math.ceil(target[i] / safe_volume[i])) for i in range(3))
    cell_size = tuple(target[i] / divisions[i] for i in range(3))

    dowel_desc = ", ".join(f"{lbl} ({mm:.3f} mm)" for lbl, mm in allowed_sizes)

    print("\n--- plan ----------------------------------------")
    print(f"  input:          {input_path}")
    print(f"  target bbox:    {target[0]:.1f} x {target[1]:.1f} x {target[2]:.1f} mm")
    print(f"  printer bbox:   {print_volume[0]:.1f} x {print_volume[1]:.1f} x {print_volume[2]:.1f} mm")
    print(f"  safety margin:  {PRINT_SAFETY_FRACTION*100:.0f}% -> "
          f"{safe_volume[0]:.1f} x {safe_volume[1]:.1f} x {safe_volume[2]:.1f} mm usable")
    print(f"  grid:           {divisions[0]} x {divisions[1]} x {divisions[2]}  "
          f"(cells ~{cell_size[0]:.1f} x {cell_size[1]:.1f} x {cell_size[2]:.1f} mm)")
    print(f"  hollow:         {hollow}")
    print(f"  dowel set:      {dowel_desc}")
    print(f"  wall markings:  {'yes (0.6 mm engraved letters)' if add_markings else 'no'}")
    print(f"  output:         {output_dir}")
    print("-------------------------------------------------")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading & cleaning mesh")
    mesh = load_and_clean(input_path)

    print("\n[2/5] Scaling to target bounding box")
    mesh = scale_to_bbox(mesh, target)

    print("\n[3/5] Cutting into grid cells (boolean intersections)")
    pieces = cut_into_pieces(mesh, divisions, cell_size)
    if not pieces:
        print("ERROR: no non-empty pieces produced. Abort.")
        sys.exit(1)

    print("\n[4/5] Drilling dowel holes + engraving wall markings")
    pieces, face_records = drill_dowel_holes(
        pieces, divisions, cell_size, allowed_sizes, add_markings=add_markings,
    )

    if hollow:
        print("      adding 4 mm vent holes for hollow printing")
        pieces = add_vent_holes(pieces)

    print("\n[5/5] Exporting STLs, assembly diagram, README")
    records = export_pieces(pieces, output_dir)
    write_assembly_map(records, divisions, cell_size, output_dir / "assembly_map.png")
    write_readme(
        records, divisions, cell_size, target, hollow,
        allowed_sizes, face_records,
        output_dir / "README.txt",
    )

    print(f"\nDone - {len(records)} piece(s) in {output_dir}")
    print(f"  see {output_dir / 'README.txt'} for print + assembly instructions.")


def interactive(input_path: Path) -> None:
    print(f"\nInput model: {input_path}")
    mesh_preview = trimesh.load(str(input_path), force="mesh")
    ext = mesh_preview.extents
    print(f"Current mesh extents: {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} mm "
          f"({ext[0]/25.4:.2f} × {ext[1]/25.4:.2f} × {ext[2]/25.4:.2f} in)")

    target = _prompt_triplet(
        "\nTarget real-world size X Y Z (mm; suffix in/cm/mm accepted)", None
    )
    print_volume = _prompt_triplet(
        "Printer build volume X Y Z (mm)", (250.0, 250.0, 250.0)
    )

    safe = tuple(v * (1.0 - PRINT_SAFETY_FRACTION) for v in print_volume)
    divisions = tuple(max(1, math.ceil(target[i] / safe[i])) for i in range(3))
    cell_size = tuple(target[i] / divisions[i] for i in range(3))
    print(f"\n-> with 10% safety margin: {safe[0]:.1f} x {safe[1]:.1f} x {safe[2]:.1f} mm")
    print(f"-> grid: {divisions[0]}x{divisions[1]}x{divisions[2]} "
          f"(cell ~{cell_size[0]:.0f}x{cell_size[1]:.0f}x{cell_size[2]:.0f} mm)")

    hollow = _prompt_yn("\nHollow pieces? (0% infill + vent hole)", default=False)
    add_markings = _prompt_yn(
        "Engrave matching letter on each pair of cut faces? (helps assembly)",
        default=True,
    )
    allowed_sizes = _prompt_dowel_set(cell_size)

    raw = input("\nOutput directory [./split_output]: ").strip()
    output_dir = Path(raw) if raw else Path("./split_output")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not _prompt_yn(f"{output_dir} exists and is not empty. Overwrite?"):
            print("aborted")
            return

    run(input_path, target, print_volume, hollow, allowed_sizes, output_dir,
        add_markings=add_markings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale, slice, and drill a 3D model for multi-part printing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Run with no arguments (or with just the input path) for interactive prompts.

            Example:
              python split_model.py gator.stl --target 2000 720 600 \\
                     --print-volume 250 250 250 --hollow --dowel-mm 9.525
        """),
    )
    parser.add_argument("input", nargs="?", help="Input STL or 3MF")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Target bbox in mm")
    parser.add_argument("--print-volume", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=[250.0, 250.0, 250.0],
                        help="Printer build volume in mm (default 250 250 250)")
    parser.add_argument("--hollow", action="store_true",
                        help="Hollow mode (0%% infill + 4 mm vent hole per piece)")
    parser.add_argument("--no-markings", action="store_true",
                        help="Do NOT engrave letter labels on cut faces "
                              "(by default, matching faces get the same letter A/B/C/... to aid assembly)")
    parser.add_argument("--dowel-sizes", type=str, default=None,
                        help=("Comma-separated dowel sizes (mm, indices 1-9 "
                              "of standard list, or fractions like 1/4,3/8,1/2). "
                              "Program auto-picks per face. Default: 1/4,3/8,1/2,5/8."))
    parser.add_argument("--dowel-mm", type=float, default=None,
                        help="DEPRECATED: single dowel size in mm. "
                              "Equivalent to --dowel-sizes <value>.")
    parser.add_argument("--output", default="./split_output", help="Output directory")
    args = parser.parse_args()

    if args.input is None:
        raw = input("Input STL/3MF path: ").strip().strip('"')
        args.input = raw

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        sys.exit(1)

    if args.target is None:
        interactive(input_path)
        return

    target = tuple(args.target)
    print_volume = tuple(args.print_volume)
    safe = tuple(v * (1.0 - PRINT_SAFETY_FRACTION) for v in print_volume)
    divisions = tuple(max(1, math.ceil(target[i] / safe[i])) for i in range(3))
    cell_size = tuple(target[i] / divisions[i] for i in range(3))

    if args.dowel_sizes is not None:
        allowed = _parse_dowel_list(args.dowel_sizes)
        if not allowed:
            print("ERROR: could not parse any sizes from --dowel-sizes")
            sys.exit(1)
    elif args.dowel_mm is not None:
        allowed = [(f"{args.dowel_mm:.2f}mm", args.dowel_mm)]
    else:
        allowed = _default_dowel_set(cell_size)

    run(input_path, target, print_volume, args.hollow, allowed, Path(args.output),
        add_markings=not args.no_markings)


if __name__ == "__main__":
    main()
