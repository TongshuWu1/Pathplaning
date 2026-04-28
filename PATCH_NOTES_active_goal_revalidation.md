# Active goal revalidation patch

This patch fixes the case where a robot commits to a Voronoi/frontier goal while it is still unknown, then later LiDAR discovers that the goal or the route is no longer valid.

## Main behavior changes

1. **Reject large goal repairs before committing**
   - A* may snap a blocked goal to a nearby free cell.
   - Tiny snaps are allowed.
   - Large snaps are rejected because they usually mean the assigned frontier/Voronoi target was inside an obstacle or behind a newly discovered wall.

2. **Revalidate active targets after every map update**
   - After each robot senses and updates its local map, the simulator checks the robot's current target and upcoming path.
   - If the target is now blocked/occupied, the target is cleared and the robot replans immediately.
   - If the target no longer has useful unknown cells nearby, the region claim is released and the robot replans.
   - If the path becomes blocked by newly observed obstacles, the robot records a temporary route block and replans.

3. **Break target commitment only when necessary**
   - Normal target commitment is preserved while the target and path are still valid.
   - New information from LiDAR can override commitment when the old plan is stale.

## Files changed

- `src/config.py`
- `src/policy.py`
- `src/simulator.py`

## New config parameters

```python
goal_repair_max_dist_m: float = 1.10
active_goal_revalidation_enabled: bool = True
active_goal_min_unknown_cells: int = 4
active_goal_check_radius_factor: float = 0.75
active_goal_path_check_points: int = 8
active_goal_replan_cooldown_s: float = 0.35
```
