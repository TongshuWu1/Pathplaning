from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


class RobotLogger:
    def __init__(self, robot_id: int, run_dir: str | None, enabled: bool = True) -> None:
        self.robot_id = robot_id
        self.enabled = enabled and bool(run_dir)
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.snapshot_path = None
        self.txt_path = None
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.snapshot_path = self.run_dir / f'robot_{robot_id}_knowledge.json'
            self.txt_path = self.run_dir / f'robot_{robot_id}_knowledge.txt'
            self.snapshot_path.write_text('{}\n', encoding='utf-8')
            self.txt_path.write_text('', encoding='utf-8')
        self._last_snapshot_time = float('-inf')


    def log(self, *args: Any, **kwargs: Any) -> None:
        return

    def write_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        now: float | None = None,
        min_period_s: float = 0.0,
        force: bool = False,
    ) -> None:
        if not self.enabled or self.snapshot_path is None or self.txt_path is None:
            return
        if now is None:
            try:
                now = float(snapshot.get('time_s', 0.0))
            except Exception:
                now = 0.0
        if not force and float(now) - self._last_snapshot_time < float(min_period_s):
            return
        self._last_snapshot_time = float(now)
        serializable = self._to_serializable(snapshot)
        self.snapshot_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        self.txt_path.write_text(self._to_text(serializable), encoding='utf-8')

    def _to_text(self, snapshot: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"robot {snapshot.get('robot_id')} knowledge @ t={snapshot.get('time_s', 0.0):.2f}")
        self_info = snapshot.get('self', {})
        lines.append('self:')
        lines.append(f"  pose_est={self_info.get('pose_est_xy')} cov_trace={self_info.get('cov_trace')} target={self_info.get('target_xy')} region={self_info.get('current_region_id')}")
        lines.append(f"  home_connected={self_info.get('home_connected')} home_hops={self_info.get('home_hops')} direct_neighbors={self_info.get('direct_neighbors')}")
        lines.append(f"  landmarks={sorted((self_info.get('landmarks') or {}).keys())}")
        teammates = snapshot.get('teammates', {})
        if not teammates:
            lines.append('teammates: none')
        else:
            lines.append('teammates:')
            for key in sorted(teammates.keys(), key=lambda x: int(x)):
                info = teammates[key]
                lines.append(
                    f"  robot {key}: pose={info.get('pose_xy')} region={info.get('current_region_id')} stale={info.get('is_stale')} age_s={info.get('age_s')} "
                    f"source={info.get('source_robot_id')} route_points={info.get('route_point_count')} recent_points={info.get('recent_point_count')} "
                    f"landmarks={sorted((info.get('landmarks') or {}).keys())}"
                )
        return '\n'.join(lines) + '\n'

    def _to_serializable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._to_serializable(v) for k, v in value.items()}
        if hasattr(value, '__dict__'):
            return {str(k): self._to_serializable(v) for k, v in value.__dict__.items()}
        return value
