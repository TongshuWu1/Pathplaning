from __future__ import annotations

import argparse
from pathlib import Path

from src.environment import load_environment
from src.ui.environment_editor import EnvironmentEditor


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
    parser = argparse.ArgumentParser(description="Pathplaning environment builder")
    parser.add_argument("--env-file", default="environment_01.json", help="JSON map path, or a name under environments/")
    parser.add_argument("--width", type=int, default=1600, help="window width")
    parser.add_argument("--height", type=int, default=950, help="window height")
    args = parser.parse_args()

    env_path = resolve_environment_path(args.env_file)
    try:
        env = load_environment(env_path)
    except FileNotFoundError:
        env = None
    EnvironmentEditor(env=env, width=args.width, height=args.height, env_dir=env_path.parent, filename=str(env_path)).run()


if __name__ == "__main__":
    main()
