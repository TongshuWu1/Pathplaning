# Search-CAGE Clean Baseline

This project is a reconstructed clean baseline for LiDAR-guided multi-robot search, mapping, route discovery, and LOS-limited reporting.

The design goal is:

> Robots do not know the finite ground-truth world or hidden target. They use EKF-like pose estimation only to estimate pose/confidence, use LiDAR to build local maps and decide motion safety, search for the hidden target, build route evidence, evaluate HOME-to-target routes, and report route certificates back to HOME through direct LOS packet exchange.

## What changed in this version

This version keeps the clean Search-CAGE backend, but restores the old simulator feel:

- old-style 30 x 30 m finite world
- rectangular HOME base region in the lower-left corner
- fewer, larger rectangular obstacles with spacing margins
- old-style spawn-clear region near HOME
- landmark beacons sampled with spacing/clearance rules
- hidden target near the far side, protected from obstacle placement
- old-style Matplotlib dashboard layout:
  - top toolbar
  - Global Truth map
  - Team Fused / HOME-Reported Belief map
  - Mission status / route table
  - compact local belief cards for all robots
- status/legend information is kept outside map panels to avoid overlap

## Run

```bash
pip install -r requirements.txt
python main.py
```

Headless run:

```bash
python main.py --headless --steps 300
python -m scripts.smoke_test
```

## UI controls

- **Start / Pause**: start or stop simulation stepping.
- **Toggle rays**: show/hide LiDAR rays.
- **Fused quality**: switch the team belief map between occupancy-only and red/green cell-quality overlay.
- **Route graph**: show/hide route graph/certificate overlays.
- **Reset**: rebuild the world using the Seed, Robots, Obstacles, and Landmarks boxes.

## Main modules

```text
src/config.py              grouped configuration
src/world.py               finite hidden-truth world + old-style generation
src/sensors.py             LiDAR sensing
src/localization.py        EKF-like pose belief
src/mapping.py             LiDAR occupancy map + quality grid
src/lidar_assessment.py    scan-map consistency and clearance assessment
src/planner.py             local grid planner
src/cage_graph.py          route graph and certificates
src/communication.py       direct LOS packet exchange and HOME memory
src/robot.py               robot agent using LiDAR-map-driven planning
src/simulator.py           orchestrator
src/ui/matplotlib_dashboard.py  old-style dashboard on new backend
```

## Design rule

A communication line means direct packet exchange is currently possible:

```text
line shown between robots/home <=> direct LOS + communication range packet edge
```

EKF does not choose targets directly. It estimates pose and confidence. LiDAR/map evidence drives movement, path invalidation, frontiers, route graph updates, and target search.
