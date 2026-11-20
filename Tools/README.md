# Spot: Tools

| Script | Purpose |
|---|---|
| `ptz_jog.py` | Move the Spot CAM PTZ camera from the keyboard (type a key, then Enter): `a`/`d` pan ±5° (`A`/`D` ±20°), `w`/`s` tilt, `+`/`-` zoom, `p` print position, `q` quit. |

```
cp ~/SPOT-code/Tools/ptz_jog.py ~/Spot_SDK/spot-sdk/python/examples/spot_cam/
cd ~/Spot_SDK/spot-sdk/python/examples/spot_cam
python3 ptz_jog.py 192.168.80.3
```
