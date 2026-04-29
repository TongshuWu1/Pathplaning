from src.config import AppConfig
from src.simulator import Simulator


def main() -> None:
    sim = Simulator(AppConfig())
    status = sim.run_headless(steps=80)
    assert sim.step_count > 0
    assert len(sim.robots) == sim.cfg.robot.count
    assert all(r.scan is not None for r in sim.robots)
    assert all(r.assessment.front_clearance >= 0 for r in sim.robots)
    # Direct communication lines and packet edges share the same state object.
    assert len(sim.comm_state.robot_segments) == len(sim.comm_state.direct_robot_edges)
    print("smoke_test_ok", status.phase, status.success)


if __name__ == "__main__":
    main()
