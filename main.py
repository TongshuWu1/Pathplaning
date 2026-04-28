from __future__ import annotations

import argparse

from src.config import AppConfig
from src.simulator import Simulator
from src.ui.matplotlib_dashboard import MatplotlibDashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Search-CAGE LiDAR-guided multi-robot baseline")
    parser.add_argument("--headless", action="store_true", help="run without UI")
    parser.add_argument("--steps", type=int, default=500, help="headless simulation steps")
    parser.add_argument("--seed", type=int, default=None, help="override world seed")
    args = parser.parse_args()

    cfg = AppConfig()
    if args.seed is not None:
        # Dataclasses are frozen; construct a replacement for the world section.
        from dataclasses import replace
        cfg = replace(cfg, world=replace(cfg.world, seed=args.seed))
    sim = Simulator(cfg)

    if args.headless:
        status = sim.run_headless(args.steps)
        print(f"phase={status.phase} success={status.success} message={status.message}")
        print(f"time_s={sim.time_s:.1f} home_target={sim.home_memory.target.detected} routes={len(sim.home_memory.best_routes)}")
        for i, route in enumerate(sim.home_memory.best_routes[:4]):
            print(f"route[{i}] length={route.length:.2f} clearance={route.min_clearance:.2f} cert={route.certificate:.2f} reported={route.reported_home} status={route.status}")
    else:
        MatplotlibDashboard(sim).run()


if __name__ == "__main__":
    main()
