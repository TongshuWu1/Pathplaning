# Cleanup Notes

This cleanup prepares the simulator for the CAGE / LiDAR-guided actionable exploration direction.

## Removed

- Cooperative rescue / teammate-assistance subsystem.
- Rescue status fields from robot state, packets, logs, and UI text.
- Helper standoff/path-planning methods from the simulator.
- Rescue-triggered covariance shrink behavior.

## Kept

- Local stuck-route recovery through temporary planning-only route blocks.
- Active target/path revalidation after new LiDAR/map updates.
- Localization recovery toward home, landmarks, and currently reachable teammate anchors.
- LOS packet sharing of pose, target, route memory, landmark beliefs, and teammate knowledge.

## Next research layer

The next major addition should be a LiDAR scan-map consistency module and route-certification/nav-graph layer:

1. `lidar_consistency.py` for scan-map agreement, clearance, open sectors, bottleneck/junction/dead-end cues.
2. `route_certification.py` for safe-route evidence from LiDAR, traversal success/failure, covariance, and connectivity.
3. `nav_graph.py` / `cage_policy.py` for graph tasks: extend safe route, resolve bottleneck, confirm dead end, close loop, certify edge, re-anchor.
