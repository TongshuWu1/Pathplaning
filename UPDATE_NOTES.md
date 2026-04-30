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


## Full patch set applied in this build

- Robot-to-robot packets no longer merge occupancy cells into each robot's local map.
  - `robot.map` remains local LiDAR evidence only.
  - `home_memory.map` remains the team-fused visualization/evaluation map.
- Robot packets now support intent-only peer packets to reduce unnecessary map traffic.
- Added same-cycle exploration reservations:
  - robots plan sequentially within a sim step;
  - already-selected teammate frontiers are immediately penalized;
  - early exploration no longer collapses all robots onto the same first frontier.
- Added robot-sector frontier approach selection:
  - each robot has a preferred outward sector;
  - frontier approach points are chosen with sector bias, clearance, and teammate-goal avoidance.
- Added lightweight LiDAR scan-to-local-map correction:
  - bounded scan matching nudges pose estimate and covariance;
  - LiDAR now helps localization instead of only mapping.
- Added exploration-complete return-home behavior:
  - when team-known coverage is high and useful frontiers remain low for a stable window,
    all robots switch to return-home behavior.
- Fixed `RouteGraph.top_routes(k)` so it can return multiple route candidates instead of only the first.
- Fixed the UI requirement:
  - button label is exactly **Fused quality**;
  - quality overlay is applied only to the HOME/Team Fused Belief map, not local maps.
- Performance tuning:
  - occupancy resolution changed to 0.30 m/cell;
  - LiDAR rays reduced to 48;
  - frontier sample count reduced to 36;
  - packet period increased to 0.8 s;
  - scan matching is throttled to keep long runs responsive.

## Localization and fused-quality correction patch

- Replaced the old pseudo landmark correction with a real EKF range/bearing landmark update.
- Added noisy simulated landmark measurements from hidden truth rather than perfect pose correction.
- Added covariance floors/limits so the estimator does not become falsely overconfident.
- Expanded LiDAR scan matching from six candidate probes to a bounded correlative local search over x/y/yaw offsets.
- Changed map insertion confidence to combine covariance, scan-map consistency, landmark anchoring, and LiDAR match confidence.
- Scored scan-map consistency before inserting the current scan, avoiding self-confirming quality.
- Cleaned fused-quality rendering so normal occupancy maps are occupancy-only; the red/green confidence overlay is only shown on HOME Fused Belief when `Fused quality` is on.

## 2026-04-29 — Knowledge-map / HOME upload / fused-quality patch

- Added separate per-robot `self_map` and `knowledge_map`.
  - `self_map`: only this robot's own LiDAR observations.
  - `knowledge_map`: own LiDAR observations plus teammate/relay map digests received through LOS communication.
  - `robot.map` remains an alias to `knowledge_map` so existing planning/UI code works.
- Robot-to-robot LOS packets now carry a bandwidth-limited knowledge-map digest.
- HOME communication now builds a HOME-connected communication component and receives full knowledge uploads from connected robots.
- Returning robots perform a throttled full HOME upload before mission completion can be reported.
- `OccupancyGrid.make_digest(..., max_cells=None)` now supports complete full-map upload.
- Fused quality remains HOME-only and is driven by mapping pose quality.
- Pose quality now directly combines position covariance, heading covariance, scan-map consistency, landmark anchoring, and LiDAR match confidence.

## Target roundtrip mission patch

- Target detection no longer ends the mission or sends everyone home immediately.
- Once any robot learns the target, every robot starts target-guided navigation/exploration.
- Robots try to reach the target from their current positions, creating different route attempts.
- When a robot reaches the target region, it switches to `RETURN_HOME_AFTER_TARGET`.
- When it returns HOME, it marks `WAIT_AT_HOME_DONE` and uploads its map/route evidence.
- HOME stores per-robot route candidates with route length, return length, mean quality, minimum clearance, and unknown fraction.
- Mission completion now requires the configured number of robots to complete target roundtrip; by default, all robots must complete it.
- Passage quality overlay remains on the HOME fused map and can be used to evaluate the reliability/clearance of the route evidence.

## Target Direct + Persistent Team Trajectories Patch

- Added persistent estimated `trajectory_from_home` for each robot.
- Added `trajectory_digest` to robot packets: full downsampled path history from HOME to current estimated pose.
- Receivers store `known_teammate_trajectories` persistently instead of expiring them with short-term intent.
- Frontier scoring now includes a soft teammate trajectory-history penalty and own trajectory-history penalty to reduce duplicate exploration without hard banning revisits.
- Local knowledge maps now draw each robot's own trajectory and known teammate trajectories.
- Added `Team paths` UI toggle, ON by default.

## Target-known trajectory-penalty fix

- In normal exploration, robots still use persistent teammate trajectory histories as a soft penalty to reduce duplicated exploration.
- Once a target position is known, robots ignore teammate path/trajectory/history penalties and prioritize moving toward the target from their current positions.
- After reaching the target, robots return HOME as before.

## Passage quality / HOME target reporting patch

- Passage quality is now a cell-wise execution/traversal score: `free_score * map_confidence * clearance_score`.
- Corridor centers score higher than cells near walls because clearance is explicitly included.
- All passage-quality weights and thresholds are labeled in `PassageQualityConfig` in `src/config.py`.
- HOME target acceptance is controlled by `TargetReportingConfig`. By default, HOME only accepts a target report from the original observing robot, not a relayed target report from another robot.

## Review follow-up fixes

- Target roundtrip reach is now credited only when the simulator confirms physical arrival at the hidden target region, not merely when estimated pose reaches the estimated target point.
- Robots that reach the estimated target point without physical confirmation keep probing locally instead of stopping.
- Direct HOME uploads now go through a HOME-connected LOS communication guard unless an explicit target-reporting config disables that requirement.
- `allow_robot_to_robot_target_share` is now honored by robot packet reception.
- Passage-quality colors are normalized per frame: the lowest current score renders full red, the highest current score renders full green, and intermediate values render along a red-yellow-green scale.

## LiDAR and exploration-target selection patch

- LiDAR raycasts now use configurable raycast step size, range-dependent hit noise, optional hit dropout, and stable max-range returns for no-hit freespace readings.
- LiDAR map insertion now splats free and hit observations through a small configurable kernel so ray maps do not leave as many unvisited one-cell gaps.
- LiDAR safety assessment now uses percentile sector clearance instead of raw minimum range, reducing one-ray noise in front/side clearance decisions.
- Open-sector selection now considers sector width, sector depth, and forward alignment instead of only choosing the widest open span.
- Frontier selection now scores LiDAR-open-direction alignment and then A* evaluates the top frontier candidates before committing a next target.
- Frontier information now combines weak local frontier size with expected LiDAR visibility from the candidate viewpoint, so robots prefer positions that can actually reveal more unseen map through the ray sensor.
- Exploration reward breakdown now exposes `frontier_gain`, `expected_visibility`, `local_info`, and `visibility_info`.
- Exploration reward breakdown now exposes `pre_plan_score`, `planned_path_length`, `planned_path_clearance`, `planned_path_unknown_fraction`, and path penalty/bonus terms.
- Traversability now treats map boundaries as inflated obstacles, so exploration targets do not get committed directly against the world edge.
- After planning, the robot's committed goal is updated to the actual reachable path endpoint returned by the planner.

## Target visibility and passage overlay patch

- Target roundtrip reach now requires physical target visibility from the simulator truth state instead of physical contact with the target region.
- Estimated-target proximity without visibility remains a probe/search condition, not a completed target reach.
- Passage-quality overlay now excludes unexplored cells from the red-green scale; unexplored cells remain the base unknown-map color.
- Passage-quality display applies a local neighbor mean over explored free cells before coloring, so the overlay is smoother without treating unknown cells as low-quality passage.

## Passage safety semantics patch

- Passage quality now means future execution-route safety from HOME to target, not mapping confidence.
- The passage score is now dominated by occupancy safety and obstacle clearance; mapper confidence is only a soft reliability discount with a configurable floor.
- Clearance scoring now uses a broader, adaptive reference clearance and smoothstep gradient, so corridor centers are highest quality and cells do not jump straight from obstacle-red to full green.
- Passage-quality display normalization uses robust explored-cell percentiles so the overlay shows mid-range safety differences instead of being dominated by outliers.
- HOME passage evaluation now asks `GridPlanner` to plan over the passage-safety score, so the reported route is biased toward the safest available corridor.
- Low-confidence explored free cells remain visible as low/medium passage quality rather than being hidden with unexplored cells.

## EKF teammate fusion / robot collision patch

- EKF localization now fuses LOS teammate relative observations with the teammate's reported pose covariance, so teammate localization can help but high teammate uncertainty weakens the correction.
- Robots now use a larger car-like body footprint in config and dashboard drawing.
- The simulator prevents robot-robot physical overlap using the configured robot radius and collision buffer.
- Local control slows and turns away from nearby teammates using received teammate pose/covariance.
- `GridPlanner` accepts dynamic teammate obstacles and adds hard keepout plus a soft cost halo, so current teammate positions and path digests affect route planning.
- Frontier scoring now includes a teammate-mapped-history likelihood from received visit/trajectory digests, reducing the chance that a robot chooses a target another teammate probably already mapped.

## Goal ownership hysteresis patch

- Frontier goal commitment now lasts longer by default.
- Robots finish the current committed goal when they are already close instead of switching because a newly scored frontier is slightly better.
- While the robot is still making progress, a replacement goal must beat the committed goal by a larger score margin.
- Blocked paths, stalled progress, reached goals, and target-known behavior still override commitment.

## Exploration target-selection major rework

- Simplified normal exploration behavior: robots now select reachable frontier goals by prioritizing unknown-space gain, expected LiDAR visibility, clearance, and distance.
- Added LiDAR-scaled teammate path avoidance: hard avoid radius defaults to 0.5 * lidar range, soft avoid radius defaults to 1.0 * lidar range.
- Added strict/relaxed exploration passes: strict mode rejects goals inside teammate-covered path corridors; relaxed mode allows overlap only when no clean reachable frontier exists.
- Added startup deployment mode (`DEPLOY_FROM_HOME`) so robots initially spread away from HOME along different launch directions before normal frontier exploration starts.
- Added same-step frontier/goal reservation for startup and frontier tasks so later-planned robots avoid picking the same initial/frontier region.
- Kept target-known and return-home workflows separate from normal exploration scoring.

Key config knobs are in `PlanningConfig`:

- `exploration_enable_startup_deployment`
- `startup_deployment_lidar_fraction`
- `startup_deployment_done_lidar_fraction`
- `startup_deployment_angle_spread_deg`
- `exploration_teammate_hard_avoid_lidar_fraction`
- `exploration_teammate_soft_avoid_lidar_fraction`
- `exploration_frontier_reservation_lidar_fraction`
- `exploration_unknown_gain_weight`
- `exploration_visibility_gain_weight`
- `exploration_clearance_weight`
- `exploration_distance_weight`
- `exploration_teammate_path_penalty_weight`

## Next-best-view exploration cleanup patch

- Normal exploration no longer selects the next target directly from frontier clusters.
- Robots now choose a **next-best-view scan pose**: a reachable candidate cell, often inside/just beyond the unexplored area, scored by expected LiDAR unknown-cell gain.
- Frontiers are kept only for target-guided fallback behavior and UI/frontier counting.
- The normal exploration task name is now `SEARCH_NBV`.
- Candidate targets are allowed in unknown space if the local map does not believe they are occupied and A* can reach them.
- Candidate generation is lattice-sampled for performance, with a robot-specific phase offset so robots do not all score exactly the same cells at startup.
- Strict mode rejects candidates inside the teammate LiDAR-covered path corridor; relaxed mode allows overlap only when no clean NBV target is reachable.
- Same-step reservations now include `SEARCH_NBV` goals.
- `PlanningConfig` was cleaned up: old exploration reward-soup weights were removed. The main remaining exploration knobs are:
  - `startup_deployment_enabled`
  - `startup_deployment_lidar_fraction`
  - `startup_deployment_angle_spread_deg`
  - `nbv_sample_stride_cells`
  - `nbv_max_candidates`
  - `nbv_plan_eval_count`
  - `nbv_local_unknown_radius_lidar_fraction`
  - `nbv_teammate_hard_avoid_lidar_fraction`
  - `nbv_teammate_soft_avoid_lidar_fraction`
  - `nbv_own_path_avoid_lidar_fraction`
  - `nbv_reservation_lidar_fraction`

## LOS-aware hierarchical NBV enhancement

- Added a coarse-to-fine exploration layer inspired by hierarchical planners such as TARE.
- The robot still uses the existing local NBV scan-pose selector, but now first selects a coarse unknown region from its own communication-limited `knowledge_map`.
- Coarse region selection is LOS-realistic: teammate region reservations are only known when received through the existing robot packet system.
- Robots announce their current coarse region in packets, so connected teammates can avoid duplicating the same region without using global truth.
- Added region commitment/hysteresis so a robot keeps working the same coarse region unless a much better region appears or local NBV cannot find a reachable scan pose there.
- Increased obstacle safety margins:
  - `lidar.blocked_forward_distance` from `0.62` to `0.80` m.
  - A* `planning.inflation_radius_m` from `0.72` to `0.88` m.
  - A* `planning.critical_clearance_m` from `0.56` to `0.70` m.
  - A* `planning.desired_clearance_m` from `1.10` to `1.35` m.
  - A* `planning.clearance_cost_weight` from `4.5` to `6.0`.
  - exploration approach minimum clearance from `0.60` to `0.78` m.
