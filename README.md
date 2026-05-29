# COLMAP → poses.csv / config.yaml

Convert a COLMAP reconstruction into two outputs:

- **`poses.csv`** — flat per-image camera poses (`colmap_to_poses.py`)
- **`config.yaml`** — camera calibration + metadata config (`colmap_to_config.py`)

## Files

| File | Description |
|------|-------------|
| `colmap_to_poses.py` | Pose conversion script (only needs `numpy`). |
| `colmap_to_config.py` | Calibration YAML generator (only needs `numpy`). |
| `images.txt` / `images.bin` | COLMAP per-image poses (two encodings of the same data). |
| `cameras.bin` | COLMAP intrinsics + distortion (per camera). |
| `rigs.bin` | COLMAP camera→rig extrinsics (newer rig format). |
| `poses.csv` | The generated poses (589 images). |
| `config.yaml` | The generated calibration config. |

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
| world → camera | `--raw` | `R` (as in COLMAP) | `tx,ty,tz` = translation `t` |
| camera → world | *(default)* | `Rᵀ` (quaternion conjugate) | `tx,ty,tz` = camera center `C = -Rᵀ·t` |

They describe the **same poses**, just in opposite directions.

## Output format

`poses.csv` has one row per image, sorted by `image_id`:

```
image_id, qw, qx, qy, qz, tx, ty, tz, camera_params_id, image_name
```

- `image_id` — COLMAP image id
- `qw, qx, qy, qz` — orientation quaternion
- `tx, ty, tz` — translation `t` in `--raw` mode, or camera center `C` in default mode
- `camera_params_id` — COLMAP camera id this image was taken with
- `image_name` — image filename (e.g. `DJI_0001.JPG`)

> **Note:** COLMAP only includes successfully registered images, so the
> sequence may skip frames (e.g. `DJI_0003` → `DJI_0005`). This is expected.

## Camera config YAML

`colmap_to_config.py` reads COLMAP's `cameras.bin` (intrinsics + distortion)
and, if present, `rigs.bin` (camera→rig extrinsics) and emits a calibration
YAML config:

```bash
# Point at the sparse dir (finds cameras.bin / rigs.bin) or cameras.bin directly
python colmap_to_config.py path/to/sparse -o config.yaml

# Remap camera ids to 0-based; override placeholder metadata
python colmap_to_config.py path/to/sparse --zero-based --frequency 30 \
    --initial-pose-type UNKNOWN
```

### What maps from COLMAP, and what doesn't

| Config field | Source |
|--------------|--------|
| `image_width` / `image_height` | ✅ `cameras.bin` |
| `intrinsic` (3×3) | ✅ `cameras.bin` params |
| `distortion_coefficients` | ✅ `cameras.bin` (model-dependent, OpenCV order) |
| `extrinsic` (camera→rig) | ✅ `rigs.bin` (identity for a single-sensor rig) |
| `initial_pose_type` | ⚠️ config choice — written as `UNKNOWN` placeholder |
| `frequency` | ⚠️ capture rate — not in COLMAP, placeholder |
| `sensor_name` | ⚠️ COLMAP only has numeric ids — placeholder |
| `session_camera_params_id_mapping` | ⚠️ your own grouping — placeholder |
| `stereo_pair` | ⚠️ inferred baseline hints only — commented out |

Fields that have no source in a COLMAP reconstruction are emitted as inline
`# TODO` placeholders rather than fabricated values.

**Distortion models:** PINHOLE → empty; `SIMPLE_RADIAL`/`RADIAL`/`OPENCV`/
`FULL_OPENCV` → OpenCV `[k1, k2, p1, p2, …]` order. Fisheye / FOV / thin-prism
models are emitted as raw COLMAP params with a `# WARNING`, since they don't map
onto the OpenCV pinhole distortion convention.

**Options:** `--zero-based` remaps COLMAP camera ids to 0-based (default keeps
COLMAP's native ids); `--frequency` and `--initial-pose-type` override the
metadata placeholders.
