#!/usr/bin/env python3
"""Convert a COLMAP images.txt / images.bin to a poses.csv.

COLMAP stores the WORLD->CAMERA transform per image:
    X_cam = R(qw,qx,qy,qz) @ X_world + (tx,ty,tz)

By default this script writes the CAMERA->WORLD pose, which is usually what
you want ("where the camera is and how it's oriented"):
    camera center  C   = -R^T @ t
    camera->world rot  = R^T   (output as quaternion qw,qx,qy,qz)

Use --raw to instead dump COLMAP's raw world->camera values unchanged.

Output columns:
    name, qw, qx, qy, qz, x, y, z
"""
import argparse
import csv
import struct
import numpy as np


def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def read_images_txt(path):
    """Yield (name, qw, qx, qy, qz, tx, ty, tz) for each image."""
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    # Two lines per image: pose line, then POINTS2D line (which we skip).
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        qw, qx, qy, qz = map(float, p[1:5])
        tx, ty, tz = map(float, p[5:8])
        name = p[9]
        yield name, qw, qx, qy, qz, tx, ty, tz


def read_images_bin(path):
    """Yield (name, qw, qx, qy, qz, tx, ty, tz) for each image."""
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            struct.unpack("<i", f.read(4))[0]  # image_id
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", f.read(56))
            struct.unpack("<i", f.read(4))[0]  # camera_id
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_pts)  # skip POINTS2D (x, y, point3D_id)
            yield name.decode(), qw, qx, qy, qz, tx, ty, tz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="images.txt or images.bin")
    ap.add_argument("-o", "--output", default="poses.csv")
    ap.add_argument("--raw", action="store_true",
                    help="write raw COLMAP world->camera values (no inversion)")
    args = ap.parse_args()

    reader = read_images_bin if args.input.endswith(".bin") else read_images_txt
    rows = []
    for name, qw, qx, qy, qz, tx, ty, tz in reader(args.input):
        if args.raw:
            rows.append([name, qw, qx, qy, qz, tx, ty, tz])
        else:
            R = quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            R_c2w = R.T
            C = -R_c2w @ t
            cqw, cqx, cqy, cqz = rotmat_to_quat(R_c2w)
            rows.append([name, cqw, cqx, cqy, cqz, C[0], C[1], C[2]])

    rows.sort(key=lambda r: r[0])
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "qw", "qx", "qy", "qz", "x", "y", "z"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} poses to {args.output}")


if __name__ == "__main__":
    main()
