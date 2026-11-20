# Spot: LiDAR

Reads Spot's **Velodyne Puck Hi-Res** LiDAR (16 beams, 360°, about ±3 cm) through the Spot SDK's `PointCloudClient`. No extra installs (no PyQt). Spot doesn't move and no lease is taken, so it can run next to the gesture programs.

| File | What it does |
|---|---|
| `spot_lidar_view.py` | Live top-down view (Spot in the centre, front up, rings every 1 m, colour = height), prints the nearest thing straight ahead and overall, and saves scans. |

## Running
```
cp ~/SPOT-code/LiDAR/spot_lidar_view.py ~/Spot_SDK/spot-sdk/python/examples/spot_cam/
cd ~/Spot_SDK/spot-sdk/python/examples/spot_cam
source ~/spot_venv/bin/activate
python3 spot_lidar_view.py 192.168.80.3 --list        # which point-cloud services/sources Spot has
python3 spot_lidar_view.py 192.168.80.3               # live view
python3 spot_lidar_view.py 192.168.80.3 --record 30   # save every scan for 30 s
```
Keys: `q` quit, `+`/`-` zoom, `s` save the current scan.

## Saved files (`lidar_scans/`, or `--out FOLDER`)

- **`s` key:** `scan_<time>.ply` (open in CloudCompare, MeshLab or Open3D) and `scan_<time>.npy` (N × 3, metres, Spot's body frame: x forward, y left, z up).
- **`--record`:** `recording_<time>.npz` (every scan as `points_<i>`, plus `odom_T_body` = where Spot was at each scan, and `times`) and `recording_<time>_merged_odom.ply` (all scans placed in the odometry frame: a rough 3D map if Spot moved during the recording; it drifts with distance since it's odometry, not SLAM).

```python
import numpy as np
d = np.load("lidar_scans/recording_XXXX.npz")
scan0 = d["points_0"]          # first scan, N x 3
poses = d["odom_T_body"]       # Spot's pose at each scan (4 x 4)
```

If `--list` shows **NONE**, the LiDAR payload isn't running or registered (check it's powered and on the tablet's payload page).
