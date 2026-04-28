# ME435 Rewrite Baseline

A cleaner baseline for decentralized multi-robot exploration.

## What this version does

- 2D world with exact rectangle obstacles, fixed landmarks, and a permanent home marker
- explicit **home base** in the bottom-left corner that also acts as a trusted localization marker
- continuous robot collision against true obstacle geometry
- per-robot local occupancy maps built only from that robot's LiDAR
- separate **Team Fused Belief** map built from shareable robot maps, not global truth
- per-cell fused confidence / uncertainty using pose covariance, sensor range, evidence type, and relay hops
- winner-take-best fusion: each cell uses the robot source with the highest confidence
- communication-respecting fusion: disconnected robots do not instantly update the shared team map
- lightweight teammate packets relayed through the current communication graph
- teammate packets now share estimated pose + covariance, full remembered estimated path from home, and navigation keypoints (no true-path leakage)
- landmark-based EKF-style localization updates from fixed landmarks and the home marker
- teammate-assisted cooperative localization using received teammate mean/covariance as uncertain mobile references
- decentralized frontier exploration with teammate-trace avoidance
- **phase 1 navigation hardening**
  - larger planner clearance
  - no diagonal corner cutting in A*
  - exact-geometry path compression checks
  - blocked / stuck detection with waypoint skipping and forced replanning
- **phase 2 LOS communication graph**
  - robot-to-robot links require line of sight and communication range
  - home-base links require LOS to the home region and communication range
  - multi-hop connectivity is tracked every step
  - robots only receive teammate packets from their reachable communication component
- modeling consistency fixes:
  - LiDAR mapping now consumes sensor-frame range/bearing beams and projects them from the robot's estimated pose
  - teammate path sharing now uses estimated history rather than ground-truth trajectory
  - each robot keeps a remembered teammate path/keypoint overlay on its own local map, even after LOS is lost
- additional speed work:
  - faster grid indexing in mapping
  - cached planner inflation offsets
  - one-shot reachable-set computation for frontier scoring instead of repeated trial planning
- GUI with:
  - global truth panel
  - separate Team Fused Belief panel
  - estimated robot positions and known landmarks on the fused belief panel
  - one local-map panel per robot
  - own path/target overlays
  - received teammate paths / positions / targets
  - LOS robot links and home links on the global panel
  - local cards showing whether each robot is home-connected
- control bar with:
  - Start
  - Stop
  - Reset
  - Seed
  - Robot count
  - Obstacle count
  - Landmark count
  - ray toggle
  - **Fused quality** toggle for green/yellow/red confidence overlay on the Team Fused Belief map

## What this version intentionally leaves simple

- localization is still intentionally lightweight rather than a full multi-robot SLAM backend
- no role-switch logic yet
- no return-home mission logic yet

This is still a baseline, but it now has the first visualization-first team belief / uncertainty-aware mapping layer. Planning still uses each robot local map; the fused map is ready for later planning/evaluation integration.

## Run

```bash
python main.py
```

## Notes

- The simulation starts **paused** by default.
- Changing the text-box values takes effect when you press **Reset**.
- Reset regenerates the world and robot roster from the current controls.
