# Pathplaning Environment Editor

Fresh project foundation for building custom robot planning environments.

The active code now focuses only on environment authoring:

- draw custom rectangular obstacles
- draw thin walls for corridors
- create room outlines with configurable door side
- place landmark beacons for future localization/sensing experiments
- preview the robot body and safety footprint at map scale
- place HOME and target
- select, move, erase, save, and load maps
- store maps as JSON under `environments/`

The old simulator, robot logic, planner, dashboards, and notes were moved to:

```text
archives/legacy_baseline_source_2026-05-05/
archives/pathplaning_archive_2026-05-05_current_baseline.tar.gz
```

## Run

```bash
pip install -r requirements.txt
python launch_envbuild.py
python launch_sim.py
```

`main.py` remains as a compatibility shortcut for `launch_envbuild.py`.

Open a specific map in either app:

```bash
python launch_envbuild.py --env-file environment_01.json
python launch_sim.py --env-file /absolute/path/to/my_map.json
```

The included starter map is `research_floorplan_01`: a 34 x 26 m indoor layout
with rooms, corridors, door gaps, obstacles, eight landmarks, HOME, and a target
room. The default robot preview is a smaller `0.72 x 0.42 m` body with a
`0.14 m` safety margin. Map geometry is locked to a `0.25 m` world grid.

## Editor Controls

- Select: select and drag existing objects.
- Obstacle: drag solid rectangular blocks.
- Wall: drag thin horizontal or vertical wall segments.
- Room: drag a room outline; the editor creates perimeter walls with the current door side.
- Landmark: click to add a named beacon.
- Erase / Delete: remove obstacle and wall objects.
- Home: drag a HOME region.
- Target: click to place the target marker.
- Save: write the current JSON file.
- Open... / Save As... / Rename...: use the native filesystem dialog for map files.
- Grid lock: snapping is always on for walls, obstacles, rooms, HOME, target, and landmarks.
- Door: cycle the door side for new rooms.
- Robot preview: toggle the robot footprint overlay.
- Bot +/- and Safe +/-: adjust the saved robot size and clearance margin preview.

## Simulator Controls

- Open Map...: choose a different environment JSON with the filesystem dialog.
- Reload: reload the current environment file from disk.
- WASD / arrow keys: manually drive the robot.
- Space: pause/run stepping.
- Ctrl+O: open an environment file.
- Ctrl+R: reload the current environment file.
- R: reset robot to HOME.
- Esc: quit.

The current simulator is intentionally minimal: it loads the editor JSON,
places one robot at HOME, renders the same map, and blocks motion when the
robot collision radius would hit a wall or obstacle. Planning, LiDAR, and
multi-robot behavior will build on this fresh runtime.

## Active Files

```text
main.py
launch_envbuild.py
launch_sim.py
src/environment.py
src/geometry.py
src/simulation.py
src/ui/environment_editor.py
src/ui/simulation_viewer.py
environments/environment_01.json
```
