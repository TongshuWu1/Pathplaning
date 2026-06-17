from __future__ import annotations

import argparse
from pathlib import Path

from src.environment import load_environment
from src.simulation import SimulatorState
from src.ui.simulation_viewer import SimulationViewer


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / "environments"


def resolve_environment_path(env_file: str) -> Path:
    requested = Path(env_file).expanduser()
    if requested.is_absolute():
        return requested
    if requested.parent != Path("."):
        return PROJECT_ROOT / requested
    return ENV_DIR / requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathplaning simulator")
    parser.add_argument("--env-file", default="environment_01.json", help="JSON map path, or a name under environments/")
    parser.add_argument("--width", type=int, default=1600, help="window width")
    parser.add_argument("--height", type=int, default=950, help="window height")
    parser.add_argument("--fps", type=int, default=60, help="target frames per second")
    args = parser.parse_args()

    env_path = resolve_environment_path(args.env_file)
    env = load_environment(env_path)
    sim = SimulatorState.from_environment(env, source_path=env_path)
    SimulationViewer(sim, width=args.width, height=args.height, fps=args.fps).run()


if __name__ == "__main__":
    main()
