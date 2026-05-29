#!/usr/bin/env python3
"""Generate a camera-calibration YAML config from a COLMAP reconstruction.

Reads COLMAP's `cameras.bin` (intrinsics + distortion) and, if present,
`rigs.bin` (camera->rig extrinsics) and emits the YAML config consumed
downstream. Fields that genuinely have no source in a COLMAP reconstruction
(capture frequency, sensor names, session grouping, pose-init type, stereo
pairing) are written as TODO placeholders.

What maps from COLMAP, and what does not:

    image_width / image_height   <- cameras.bin            (exact)
    intrinsic (3x3)              <- cameras.bin params      (exact)
    distortion_coefficients      <- cameras.bin params      (model-dependent)
    extrinsic (camera->rig)      <- rigs.bin                (identity for a
                                                             single-sensor rig)
    frequency                    -- not in COLMAP           (TODO)
    sensor_name                  -- not in COLMAP           (TODO, numeric id only)
    session_camera_params_id_mapping -- not in COLMAP       (TODO)
    initial_pose_type            -- downstream config       (TODO)
    stereo_pair                  -- inferred hint only       (TODO, see comments)

Usage:
    python colmap_to_config.py <sparse_dir | cameras.bin> -o config.yaml
    python colmap_to_config.py .../sparse --zero-based --frequency 30
"""
import argparse
import os
import struct
import numpy as np


# COLMAP camera model id -> (name, num_params, param_names)
# param order follows COLMAP's src/colmap/sensor/models.h
MODELS = {
    0:  ("SIMPLE_PINHOLE", 3,  ["f", "cx", "cy"]),
    1:  ("PINHOLE",        4,  ["fx", "fy", "cx", "cy"]),
    2:  ("SIMPLE_RADIAL",  4,  ["f", "cx", "cy", "k"]),
    3:  ("RADIAL",         5,  ["f", "cx", "cy", "k1", "k2"]),
    4:  ("OPENCV",         8,  ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]),
    5:  ("OPENCV_FISHEYE", 8,  ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]),
    6:  ("FULL_OPENCV",    12, ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
                                "k3", "k4", "k5", "k6"]),
    7:  ("FOV",            5,  ["fx", "fy", "cx", "cy", "omega"]),
    8:  ("SIMPLE_RADIAL_FISHEYE", 4, ["f", "cx", "cy", "k"]),
    9:  ("RADIAL_FISHEYE", 5,  ["f", "cx", "cy", "k1", "k2"]),
    10: ("THIN_PRISM_FISHEYE", 12, ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
                                    "k3", "k4", "sx1", "sy1"]),
}

# Models whose distortion maps cleanly onto OpenCV pinhole [k1,k2,p1,p2,(k3..)].
_PINHOLE_DISTORTION = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                       "OPENCV", "FULL_OPENCV"}


def read_cameras_bin(path):
    """Return {camera_id: {"model","width","height","p"}} where p maps name->value."""
    cams = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id, model_id = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            name, nparams, pnames = MODELS[model_id]
            vals = struct.unpack("<%dd" % nparams, f.read(8 * nparams))
            cams[cam_id] = {
                "model": name,
                "width": width,
                "height": height,
                "p": dict(zip(pnames, vals)),
            }
    return cams


def read_rigs_bin(path):
    """Return {ref_camera_id: {sensor_camera_id: (qw,qx,qy,qz,tx,ty,tz) cam->rig}}.

    COLMAP stores `sensor_from_rig` (rig->camera) per non-reference sensor; we
    invert it to camera->rig so it matches the YAML's "camera to vehicle"
    convention. The reference sensor is the rig origin (identity).
    """
    rigs = {}
    with open(path, "rb") as f:
        num_rigs = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_rigs):
            _rig_id = struct.unpack("<I", f.read(4))[0]
            num_sensors = struct.unpack("<I", f.read(4))[0]
            ref_type, ref_id = struct.unpack("<iI", f.read(8))
            rig = {ref_id: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}  # ref = identity
            for _ in range(num_sensors - 1):
                _stype, sid = struct.unpack("<iI", f.read(8))
                has_pose = struct.unpack("<?", f.read(1))[0]
                if has_pose:
                    q = struct.unpack("<4d", f.read(32))   # sensor_from_rig rot (wxyz)
                    t = struct.unpack("<3d", f.read(24))   # sensor_from_rig trans
                    rig[sid] = _invert_pose(q, t)
                else:
                    rig[sid] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rigs[ref_id] = rig
    return rigs


def _invert_pose(q, t):
    """Invert a (quat wxyz, trans) pose. Returns (qw,qx,qy,qz,tx,ty,tz)."""
    qw, qx, qy, qz = q
    n = qw * qw + qx * qx + qy * qy + qz * qz
    # conjugate / norm^2 = inverse quaternion
    iqw, iqx, iqy, iqz = qw / n, -qx / n, -qy / n, -qz / n
    R_inv = _quat_to_rotmat(iqw, iqx, iqy, iqz)
    t_inv = -R_inv @ np.array(t)
    return (iqw, iqx, iqy, iqz, t_inv[0], t_inv[1], t_inv[2])


def _quat_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def intrinsic_matrix(cam):
    """Return the 9 row-major values [fx,0,cx, 0,fy,cy, 0,0,1]."""
    p = cam["p"]
    fx = p.get("fx", p.get("f"))
    fy = p.get("fy", p.get("f"))
    return [fx, 0, p["cx"], 0, fy, p["cy"], 0, 0, 1]


def distortion_coeffs(cam):
    """Return (coeffs_list, note). coeffs in OpenCV [k1,k2,p1,p2,(k3..)] order."""
    model, p = cam["model"], cam["p"]
    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        return [], None
    if model == "SIMPLE_RADIAL":
        return [p["k"], 0.0, 0.0, 0.0], None
    if model == "RADIAL":
        return [p["k1"], p["k2"], 0.0, 0.0], None
    if model == "OPENCV":
        return [p["k1"], p["k2"], p["p1"], p["p2"]], None
    if model == "FULL_OPENCV":
        return [p["k1"], p["k2"], p["p1"], p["p2"],
                p["k3"], p["k4"], p["k5"], p["k6"]], None
    # Fisheye / FOV / thin-prism: NOT OpenCV pinhole distortion. Emit raw params
    # with a warning so it is never silently misinterpreted.
    raw = [v for k, v in p.items() if k not in ("fx", "fy", "f", "cx", "cy")]
    return raw, ("%s is not a pinhole-distortion model; coefficients are raw "
                 "COLMAP params, NOT OpenCV [k1,k2,p1,p2]." % model)


def fmt_num(x):
    """Compact number formatting: ints stay int-like, floats trimmed."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return repr(round(x, 10)) if isinstance(x, float) else str(x)


def fmt_list(vals):
    return "[" + ", ".join(fmt_num(v) for v in vals) + "]"


def build_yaml(cams, rigs, id_map, frequency, pose_type):
    L = []
    L.append("# Generated from COLMAP by colmap_to_config.py")
    L.append("# Fields marked TODO have no source in a COLMAP reconstruction.")
    L.append("")
    L.append("# Pose type for initialization (optional, default: EGO_MOTION)")
    L.append("initial_pose_type: %s  # TODO: EGO_MOTION | IDENTITY | UNKNOWN" % pose_type)
    L.append("")

    # All COLMAP cameras here share resolution only if equal; warn otherwise.
    widths = {c["width"] for c in cams.values()}
    heights = {c["height"] for c in cams.values()}
    any_cam = next(iter(cams.values()))
    L.append("# Image dimensions - all cameras must use the same resolution")
    if len(widths) > 1 or len(heights) > 1:
        L.append("# WARNING: COLMAP cameras differ in resolution; using the first.")
    L.append("image_width: %d" % any_cam["width"])
    L.append("image_height: %d" % any_cam["height"])
    L.append("frequency: %d  # TODO: capture rate, not in COLMAP" % frequency)
    L.append("")

    all_ids = ", ".join(str(id_map[c]) for c in sorted(cams))
    L.append("# Session to camera params ID mapping (required)")
    L.append("# TODO: split your captures into sessions; all cameras listed under one.")
    L.append("session_camera_params_id_mapping:")
    L.append("  session_0: %s" % all_ids)
    L.append("")

    L.append("# Camera parameters (required)")
    L.append("camera_params:")
    for cid in sorted(cams):
        cam = cams[cid]
        out_id = id_map[cid]
        # extrinsic from rig if available, else identity
        extr = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for rig in rigs.values():
            if cid in rig:
                extr = rig[cid]
                break
        coeffs, note = distortion_coeffs(cam)
        L.append("  %d:" % out_id)
        L.append("    sensor_name: camera_%d  # TODO: rename (COLMAP id %d, model %s)"
                 % (cid, cid, cam["model"]))
        L.append("    # Extrinsic: [QW, QX, QY, QZ, TX, TY, TZ] - camera to rig/vehicle")
        if extr == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
            L.append("    extrinsic: %s  # identity (reference / single-sensor rig)"
                     % fmt_list(extr))
        else:
            L.append("    extrinsic: %s" % fmt_list(extr))
        L.append("    # Intrinsic: 3x3 [fx,0,cx, 0,fy,cy, 0,0,1]")
        L.append("    intrinsic: %s" % fmt_list(intrinsic_matrix(cam)))
        if note:
            L.append("    # WARNING: %s" % note)
        L.append("    distortion_coefficients: %s" % fmt_list(coeffs))
    L.append("")

    # Stereo pairs: cannot be known from intrinsics. Offer baseline hints when a
    # rig defines >=2 camera positions, otherwise just a placeholder.
    L.append("# Stereo pair configuration (optional)")
    L.append("# Each entry: [left_camera_id, right_camera_id, baseline_meters]")
    centers = _rig_centers(rigs, id_map)
    if len(centers) >= 2:
        L.append("# TODO: pick real pairs. Pairwise baselines (rig frame) as a hint:")
        items = sorted(centers.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, ca), (b, cb) = items[i], items[j]
                d = float(np.linalg.norm(np.array(ca) - np.array(cb)))
                L.append("#   [%d, %d, %s]" % (a, b, fmt_num(round(d, 6))))
    else:
        L.append("# TODO: no rig with multiple cameras found (single camera).")
    L.append("# stereo_pair:")
    L.append("#   - [0, 1, 0.15]")
    return "\n".join(L) + "\n"


def _rig_centers(rigs, id_map):
    """camera center (-R^T t-style already inverted to cam->rig) per out-id."""
    centers = {}
    for rig in rigs.values():
        for cid, (qw, qx, qy, qz, tx, ty, tz) in rig.items():
            centers[id_map.get(cid, cid)] = (tx, ty, tz)
    return centers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="COLMAP sparse dir, or path to cameras.bin")
    ap.add_argument("-o", "--output", default="config.yaml")
    ap.add_argument("--zero-based", action="store_true",
                    help="remap camera ids to 0-based (default: keep COLMAP ids)")
    ap.add_argument("--frequency", type=int, default=30,
                    help="capture frequency to write (placeholder; default 30)")
    ap.add_argument("--initial-pose-type", default="UNKNOWN")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        cameras_path = os.path.join(args.input, "cameras.bin")
        rigs_path = os.path.join(args.input, "rigs.bin")
    else:
        cameras_path = args.input
        rigs_path = os.path.join(os.path.dirname(args.input), "rigs.bin")

    cams = read_cameras_bin(cameras_path)
    rigs = read_rigs_bin(rigs_path) if os.path.exists(rigs_path) else {}

    if args.zero_based:
        id_map = {cid: i for i, cid in enumerate(sorted(cams))}
    else:
        id_map = {cid: cid for cid in cams}

    yaml = build_yaml(cams, rigs, id_map, args.frequency, args.initial_pose_type)
    with open(args.output, "w") as f:
        f.write(yaml)
    print("Wrote %s (%d camera(s), %d rig(s))"
          % (args.output, len(cams), len(rigs)))


if __name__ == "__main__":
    main()
