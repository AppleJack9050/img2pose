# COLMAP → poses.csv

Convert a COLMAP reconstruction (`images.txt` / `images.bin`) into a flat
`poses.csv` of per-image camera poses.

## Files

| File | Description |
|------|-------------|
| `colmap_to_poses.py` | Conversion script (only needs `numpy`). |
| `images.txt` / `images.bin` | COLMAP output (two encodings of the same data). |
| `poses.csv` | The generated poses (589 images). |

## Usage

```bash
# Raw world->camera, COLMAP's native values (what poses.csv currently uses)
python colmap_to_poses.py images.bin --raw -o poses.csv

# Camera->world: camera position + orientation in world space
python colmap_to_poses.py images.bin -o poses.csv
```

The `.txt` and `.bin` inputs are interchangeable — they produce identical
output.

## Pose conventions

COLMAP stores the **world → camera** transform per image:

```
X_cam = R(qw,qx,qy,qz) · X_world + t(tx,ty,tz)
```

So `(tx,ty,tz)` is **not** the camera position. The two modes differ:

| Mode | Flag | Rotation | Last 3 columns |
|------|------|----------|----------------|
| world → camera | `--raw` | `R` (as in COLMAP) | translation `t` |
| camera → world | *(default)* | `Rᵀ` (quaternion conjugate) | camera center `C = -Rᵀ·t` |

They describe the **same poses**, just in opposite directions.

## Output format

`poses.csv` has one row per image, sorted by image name:

```
name, qw, qx, qy, qz, x, y, z
```

- `name` — image filename (e.g. `DJI_0001.JPG`)
- `qw, qx, qy, qz` — orientation quaternion
- `x, y, z` — translation `t` in `--raw` mode, or camera center `C` in default mode

> **Note:** COLMAP only includes successfully registered images, so the
> sequence may skip frames (e.g. `DJI_0003` → `DJI_0005`). This is expected.
