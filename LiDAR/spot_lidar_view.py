#!/usr/bin/env python3
"""
============================================================================
spot_lidar_view.py  —  Live view of Spot's LiDAR (Velodyne) point cloud
============================================================================
A simple replacement for the SDK's velodyne_client example that needs no
extra installs (no PyQt): it reads the point cloud with the Spot SDK's
PointCloudClient and draws a top-down view with OpenCV.

  * Spot is in the centre, its FRONT points UP in the window.
  * Rings every 1 m. Points are coloured by height (blue low -> red high).
  * The terminal prints, about once a second: number of points, the nearest
    thing straight ahead (±15°, at body height) and the nearest thing overall.
Spot does not move; no lease is taken, so it can run next to the gesture programs.

Usage (spot_venv, on Spot's Wi-Fi):
    python3 spot_lidar_view.py 192.168.80.3
    python3 spot_lidar_view.py 192.168.80.3 --list        # just list point-cloud services/sources
    python3 spot_lidar_view.py 192.168.80.3 --record 30   # save every scan for 30 s
Keys: q quit, + / - zoom, s save the current scan

Saving (folder lidar_scans/ by default, --out to change):
  * s key: one scan as scan_<time>.ply (open in CloudCompare / MeshLab / Open3D)
    and scan_<time>.npy (N x 3 array, metres, Spot's body frame).
  * --record SECONDS: every scan during that time goes into one file,
    recording_<time>.npz, with:
        points_<i>   (N x 3) points of scan i in the BODY frame
        odom_T_body  (K x 4 x 4) where Spot's body was (odometry) at each scan
        times        (K,) seconds since the start
    plus recording_<time>_merged_odom.ply: all scans placed in the odometry
    frame, so walking Spot around during the recording gives a rough 3D map
    (drift grows with distance; it's odometry, not SLAM).
============================================================================
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

FRONT_HALF_DEG = 15.0          # "straight ahead" sector
BODY_Z = (-0.3, 1.5)           # heights (m, body frame) that count as obstacles / people
MIN_RANGE = 0.5                # ignore points closer than this (Spot's own body)
WIN = 700                      # window size in pixels


def connect(hostname):
    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client.point_cloud import PointCloudClient
    sdk = bosdyn.client.create_standard_sdk("SpotLidarView", [PointCloudClient])
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    robot.sync_with_directory()
    return robot


def point_cloud_services(robot):
    return sorted(name for name, stype in robot.service_type_by_name.items()
                  if stype == "bosdyn.api.PointCloudService")


def decode(pc):
    """PointCloud proto -> (N, 3) float array in the sensor frame."""
    from bosdyn.api import point_cloud_pb2
    if pc.encoding == point_cloud_pb2.PointCloud.ENCODING_XYZ_32F:
        return np.frombuffer(pc.data, dtype=np.float32).reshape(-1, 3).astype(float)
    raise ValueError(f"point cloud encoding {pc.encoding} not supported yet (only XYZ_32F); tell Claude")


def odom_T_body(source):
    """4x4 pose of Spot's body in the odometry frame at the scan time (or None)."""
    from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
    tf = get_a_tform_b(source.transforms_snapshot, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    return None if tf is None else np.array(tf.to_matrix())


def save_ply(path, points):
    """Binary little-endian PLY of x, y, z (float32): opens in CloudCompare,
    MeshLab, Open3D, Blender."""
    pts = np.asarray(points, dtype="<f4")
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\nend_header\n").encode())
        f.write(pts.tobytes())


def to_body(points, source):
    """Sensor-frame points -> Spot's body frame (x forward, y left, z up)."""
    from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
    tf = get_a_tform_b(source.transforms_snapshot, BODY_FRAME_NAME, source.frame_name_sensor)
    if tf is None:
        return points
    m = tf.to_matrix()
    return points @ m[:3, :3].T + m[:3, 3]


def draw(points, scale):
    img = np.zeros((WIN, WIN, 3), np.uint8)
    c = WIN // 2
    for r in range(1, 20):
        rad = int(r * scale)
        if rad > WIN:
            break
        cv2.circle(img, (c, c), rad, (60, 60, 60), 1)
        cv2.putText(img, f"{r} m", (c + 3, c - rad + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (120, 120, 120), 1)
    # front sector lines
    for s in (-1, 1):
        a = math.radians(FRONT_HALF_DEG) * s
        cv2.line(img, (c, c), (int(c - math.sin(a) * WIN), int(c - math.cos(a) * WIN)),
                 (0, 90, 0), 1)
    if len(points):
        u = (c - points[:, 1] * scale).astype(int)        # +y (left) -> left in the image
        v = (c - points[:, 0] * scale).astype(int)        # +x (forward) -> up
        ok = (u >= 0) & (u < WIN) & (v >= 0) & (v < WIN)
        z = np.clip((points[ok, 2] + 0.5) / 2.5, 0, 1)
        colours = cv2.applyColorMap((z * 255).astype(np.uint8).reshape(-1, 1),
                                    cv2.COLORMAP_JET).reshape(-1, 3)
        img[v[ok], u[ok]] = colours
    cv2.rectangle(img, (c - 6, c - 12), (c + 6, c + 12), (0, 255, 255), 1)   # Spot
    cv2.putText(img, "FRONT", (c - 22, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hostname")
    ap.add_argument("--service", help="Point-cloud service name (default: found automatically)")
    ap.add_argument("--source", help="Point-cloud source name (default: the first one)")
    ap.add_argument("--list", action="store_true", help="List services and sources, then exit")
    ap.add_argument("--record", type=float, default=0,
                    help="Save every scan for this many seconds (one .npz + a merged .ply)")
    ap.add_argument("--out", default="lidar_scans", help="Folder for saved scans")
    args = ap.parse_args()

    robot = connect(args.hostname)
    services = point_cloud_services(robot)
    print(f"[LIDAR] Point-cloud services on this robot: {services or 'NONE'}")
    if not services:
        sys.exit("No point-cloud service found. Is the LiDAR payload powered and registered "
                 "(check the tablet's payload / services page)?")
    service = args.service or ("velodyne-point-cloud" if "velodyne-point-cloud" in services
                               else services[0])
    client = robot.ensure_client(service)
    sources = [s.name for s in client.list_point_cloud_sources()]
    print(f"[LIDAR] Service '{service}' sources: {sources or 'NONE'}")
    if args.list:
        return
    if not sources:
        sys.exit("The service has no point-cloud sources.")
    source = args.source or sources[0]
    print(f"[LIDAR] Reading '{source}'. Window: front is UP, rings every 1 m. q quits.")

    os.makedirs(args.out, exist_ok=True)
    rec = {"scans": [], "poses": [], "times": []} if args.record > 0 else None
    t_start = time.time()
    if rec is not None:
        print(f"[LIDAR] Recording every scan for {args.record:.0f} s into {args.out}/")

    scale = WIN / 12.0                                   # pixels per metre (±6 m view)
    last_print = 0.0
    points = np.zeros((0, 3))
    try:
        while True:
            t = time.time()
            try:
                resp = client.get_point_cloud_from_sources([source])[0]
                pts = decode(resp.point_cloud)
                points = to_body(pts, resp.point_cloud.source)
                if rec is not None:
                    if time.time() - t_start <= args.record:
                        pose = odom_T_body(resp.point_cloud.source)
                        rec["scans"].append(points.astype(np.float32))
                        rec["poses"].append(np.eye(4) if pose is None else pose)
                        rec["times"].append(time.time() - t_start)
                    else:
                        save_recording(args.out, rec)
                        rec = None
            except Exception as e:
                print(f"[LIDAR] read failed: {e}")
                time.sleep(0.5)
                continue
            dt = time.time() - t

            rng = np.hypot(points[:, 0], points[:, 1])
            ang = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
            body = (points[:, 2] > BODY_Z[0]) & (points[:, 2] < BODY_Z[1]) & (rng > MIN_RANGE)
            front = body & (np.abs(ang) <= FRONT_HALF_DEG)
            if time.time() - last_print >= 1.0:
                fr = f"{rng[front].min():.2f} m" if front.any() else "-"
                if body.any():
                    i = np.argmin(np.where(body, rng, np.inf))
                    near = f"{rng[i]:.2f} m at {ang[i]:+.0f} deg"
                else:
                    near = "-"
                print(f"[LIDAR] {len(points)} points ({1000 * dt:.0f} ms)   "
                      f"straight ahead: {fr}   nearest: {near}")
                last_print = time.time()

            cv2.imshow("Spot LiDAR (top view, front up)", draw(points, scale))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("+"), ord("=")):
                scale *= 1.25
            if key == ord("-"):
                scale /= 1.25
            if key == ord("s"):
                base = os.path.join(args.out, f"scan_{time.strftime('%Y%m%d_%H%M%S')}")
                np.save(base + ".npy", points)
                save_ply(base + ".ply", points)
                print(f"[LIDAR] saved {base}.ply and .npy ({len(points)} points)")
            time.sleep(max(0.0, 0.1 - (time.time() - t)))     # ~10 Hz max
    except KeyboardInterrupt:
        pass
    finally:
        if rec is not None and rec["scans"]:
            save_recording(args.out, rec)          # quit early: keep what was recorded
        cv2.destroyAllWindows()


def save_recording(out, rec):
    base = os.path.join(out, f"recording_{time.strftime('%Y%m%d_%H%M%S')}")
    arrays = {f"points_{i}": p for i, p in enumerate(rec["scans"])}
    np.savez_compressed(base + ".npz", odom_T_body=np.array(rec["poses"]),
                        times=np.array(rec["times"]), **arrays)
    merged = [p @ T[:3, :3].T + T[:3, 3] for p, T in zip(rec["scans"], rec["poses"])]
    merged = np.concatenate(merged) if merged else np.zeros((0, 3))
    save_ply(base + "_merged_odom.ply", merged)
    print(f"[LIDAR] Recording saved: {base}.npz ({len(rec['scans'])} scans) and "
          f"{base}_merged_odom.ply ({len(merged)} points)")


if __name__ == "__main__":
    main()
