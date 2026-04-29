# model_splitter

Take any STL or 3MF, scale it to a real-world size, and slice it into grid-aligned chunks that each fit on a single 3D printer build plate — with **wooden-dowel holes drilled into every shared cut face** and **matching letter labels engraved on each pair**, so reassembly is trivial.

Originally built to print a life-size alligator for a haunted house on a 250 x 250 x 250 mm printer.

## Download

Grab the latest standalone Windows exe from the [releases page](https://github.com/findingnino/model_splitter/releases). No Python install required — double-click to run interactively, or drag an STL/3MF onto it.

## What it does

1. Loads an STL or 3MF, drops floating debris, fills minor holes
2. Scales the mesh to a target XYZ bounding box you specify (mm; suffix `cm` or `in` to convert)
3. Computes a grid based on your printer's build volume (with a 10% safety margin)
4. Voxel-pre-passes the mesh to identify which grid cells actually contain material — empty cells are skipped, no booleans wasted
5. Shows you the estimated piece count and PLA filament weight, then prompts for confirmation
6. Boolean-intersects the mesh with each occupied cell to produce the pieces
7. For each pair of adjacent pieces, picks the largest dowel size from your allowed set that still fits >=2 dowels, drills aligned holes (with ray-cast / containment depth checks so they don't punch through), and engraves a unique letter label (A, B, ... Z, AA, AB, ...) on both sides of the cut
8. Optionally adds a 4 mm vent hole per piece for "hollow" printing (0% infill)
9. Exports one STL per piece + a 3-view assembly diagram + a README with a dowel shopping list and per-face breakdown

## Quick start (interactive)

```
model_splitter.exe
```

Or with the input file pre-filled (drag-and-drop, or as the first arg):

```
model_splitter.exe gator.stl
```

It'll prompt for target size, print volume, hollow yes/no, dowel set, hole clearance, and output directory.

## Quick start (CLI)

```
model_splitter.exe gator.stl ^
  --target 1500 538 1418 ^
  --print-volume 250 250 250 ^
  --dowel-sizes 1/4,3/8,1/2,5/8 ^
  --hole-clearance 0.4 ^
  --output ./gator_pieces ^
  --yes
```

All flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--target X Y Z` | (required) | Target real-world bbox in mm |
| `--print-volume X Y Z` | `250 250 250` | Printer build volume in mm |
| `--dowel-sizes` | `1/4,3/8,1/2,5/8` | Comma-separated allowed sizes (mm, fractions, or 1-9 indices) |
| `--dowel-mm` | (deprecated) | Equivalent to `--dowel-sizes <value>` |
| `--hole-clearance` | `0.4` | mm of diametral clearance added to each hole (slip fit) |
| `--hollow` | off | Add 4 mm vent hole per piece (for 0% infill prints) |
| `--no-markings` | off | Skip the engraved letter labels |
| `-y`, `--yes` | off | Skip the piece-count confirmation prompt |
| `--output` | `./split_output` | Output directory |

## Dowel sizing

By default the program is given the four standard Home Depot / Lowes wooden dowel sizes (1/4", 3/8", 1/2", 5/8") and **picks per face**: the largest size that still fits at least 2 dowels with proper edge clearance. Tiny faces get 1/4" with 2 dowels; huge faces get 5/8" with up to 6.

Pass a single size in `--dowel-sizes` to force one size everywhere:

```
--dowel-sizes 3/8
```

Or any custom set:

```
--dowel-sizes 5,9.525,12.7,1
```

Run output includes a per-size shopping list with the cut length for each dowel (lengths vary because the depth of each hole is capped to fit the local piece geometry — no through-cuts).

## Output

```
gator_pieces/
  piece_x0_y0_z0.stl
  piece_x0_y0_z1.stl
  ...
  assembly_map.png      # three orthographic views with cell labels
  README.txt            # piece list, dowel shopping list, slicer recommendations,
                        # per-face breakdown showing which letter goes with which dowel
```

## Build from source

```
git clone https://github.com/findingnino/model_splitter
cd model_splitter
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python split_model.py
```

To rebuild the standalone exe:

```
venv\Scripts\python -m PyInstaller --onefile --name model_splitter --console ^
  --collect-submodules trimesh ^
  --collect-submodules manifold3d ^
  --collect-submodules shapely ^
  --collect-submodules matplotlib ^
  --collect-data trimesh ^
  --collect-data matplotlib ^
  split_model.py
```

The `model_splitter.spec` file in the repo captures these flags.

## How the dowel-hole depth check works

For each candidate dowel position, the program shoots a small grid of points along the cylinder surface (center + 8 perimeter samples) at 20 depths from the cut plane to the ideal depth (`1x dowel diameter` per side). It uses point-in-mesh containment to find the first depth at which **any** sample point exits the piece material — that's where the cylinder would break through. The hole is shortened to that depth minus a 2 mm safety wall.

In addition, every accepted cylinder's bounding box is recorded per piece, and any candidate cylinder that would overlap a previously-placed perpendicular cylinder is skipped. This prevents two perpendicular dowels on different faces from intersecting inside the piece (which would topologically create a through-tunnel even if neither hole alone broke through a wall).

On the 1.5 m alligator (81 pieces, 335 dowels): drilling-induced through-holes went from 216 (early prototype) to 4 (current). The 4 remaining are on very irregular pieces where the boolean engine has edge cases beyond what the depth check models.

## Known limitations

- **Axis-aligned grid only.** Cuts are XYZ planes. Doesn't try to follow natural seams (e.g., between a leg and the body). Works fine for haunted-house props and most decorative use; not optimized for high-end art reproduction.
- **Through-hole edge cases.** ~1% of dowels on highly irregular pieces (cells where the mesh fills <50% of the bounding cell) can still produce a single through-hole. See note above.
- **Slow on huge meshes.** A 1 M-vertex mesh, scaled to 1.5 m, with ~80 pieces takes ~10-20 minutes to drill on a typical desktop. The boolean intersections (manifold3d backend) dominate the runtime.
- **Hollow mode is slicer-driven.** It exports watertight pieces with one drain/vent hole each; you set 0% infill + 4 walls in your slicer. The geometry isn't actually hollowed at the mesh level.

## Releases

| Version | Highlights |
| --- | --- |
| v1.0.0 | Initial release: scale, grid cut, dowel holes, letter labels |
| v1.1.0 | Voxel pre-pass to skip empty cells; piece-count confirmation |
| v1.1.1 | `--hole-clearance` exposed as a flag (default 0.4 mm) |
| v1.1.2 | Per-piece cylinder collision check; through-holes 216 -> 4 on test gator |

## License

MIT.
