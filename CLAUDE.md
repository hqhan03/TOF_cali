# CLAUDE.md

Multi-view point clouds from a TMF8829 dToF sensor on a Fairino FR5. Plain Python, no ROS. See README.md (Korean) for the workflow, safety notes, and what has and has not been verified on hardware.

## Commands

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q     # ROS on PYTHONPATH breaks pytest
env -u PYTHONPATH .venv/bin/hhg-tof scan --config config/sim_box.yaml --name box
```

## Rules

- Never run `scan --execute`, `wrist-scan --execute`, `approach --execute`, `retrace --execute`, `python -m hist_scan scan --execute`, `python -m hist_scan range-check --execute`, `python -m hist_scan tilt-check --execute`, or construct `FR5Robot(allow_motion=True)` against the real robot unless the user asks for that specific motion. `status`, `plan`, and `solve_ik` are read-only and safe.
- All motion lives in `robot/fr5.py`, behind `READ_METHODS` / `WRITE_METHODS`. Keep it there.
- `robot/fr5_io.py` is a verbatim copy from KDC-tactile-testing-platform, validated against the SDK structs. Do not edit it.
- The simulated robot has no kinematics: its IK always succeeds. Reachability claims must come from the real controller's IK, which is read-only.
- Keep `radial` and `plane_z` distances distinct: LUT output is `plane_z`. See `sensor_model.py`.
- Real-robot values have no defaults on purpose (`config.require`). Do not add guessed ones.
