# Update notes

This version enforces the design rule that each robot plans from its own local LiDAR map, EKF pose estimate, and packet-received teammate intent only.

Main changes:

- Coarser occupancy grid resolution: 0.24 m/cell for better long-run performance.
- Clearance-aware A*: larger obstacle inflation and low-clearance centerline cost.
- Reward-based frontier/target selection:
  - information gain
  - centerline/clearance reward
  - route-extension reward
  - distance cost
  - recent-visit penalty
  - strong teammate goal penalty
  - strong teammate path-overlap penalty
- Packet sharing now includes compact path digests.
- Teammate goal/path penalties use only packet-received intent.
- Local map panels show the robot's estimated pose, not its true pose.
- Local LiDAR rays originate from the estimated pose.
- UI performance improvements: fewer rendered LiDAR rays, capped displayed path length, graph edge cap, fewer redraws.
