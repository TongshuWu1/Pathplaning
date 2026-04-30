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


## Current behavior notes

- Each robot plans from its communication-limited knowledge map: its own LiDAR map plus received teammate map digests.
- HOME builds the fused team belief map from HOME-connected robot self-map uploads.
- Robot-to-robot communication shares intent, pose, path digest, visit digest, trajectory digest, target report, route evidence, and a bandwidth-limited knowledge-map digest.
- EKF pose correction uses noisy fixed-landmark observations, LiDAR scan matching, and LOS teammate relative observations weighted by the teammate's reported pose uncertainty.
- Robots are modeled as larger car-like bodies; the simulator prevents robot-robot overlap and local control slows/turns away from nearby teammates.
- The **Passage quality** button affects only the HOME/Team Fused Belief panel.
- LiDAR uses range-dependent hit noise, stable max-range freespace readings, kernelized map updates, and percentile sector clearances for less twitchy local safety decisions.
- Target roundtrip reach is credited when a robot physically sees the target, not only when it touches the target region.
- Passage quality is safety-first for future HOME-to-target execution paths: obstacle risk and clearance dominate, mapper confidence is only a soft reliability discount.
- Passage quality uses a broad, adaptive clearance gradient so corridor/open-space centers are highest quality instead of all non-wall cells immediately turning green.
- Passage quality colors are scaled over explored free cells only; unexplored cells remain visually separate from low-quality known passage.
- Exploration frontier goals are chosen from scored candidates after checking planned path length, path clearance, unknown fraction, expected LiDAR-visible unknown area, LiDAR alignment, teammate reservations, current teammate paths, and teammate mapped-history likelihood.
- Robots use goal-ownership hysteresis: once a frontier target is committed, they keep it longer, finish it when close, and only switch to a clearly better candidate unless blocked or stalled.
- Exploration now uses same-step reservations and sector-biased frontier approaches to reduce robot clustering.
- When exploration appears complete from team-known coverage and low remaining frontier count, robots switch to return-home behavior.

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
