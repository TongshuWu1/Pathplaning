"""Robot agent for the clean Search-CAGE baseline.

Every robot plans from its own communication-limited knowledge map, its EKF
pose estimate, and packet-received teammate intent.  Ground truth is used only
by the simulator to produce sensing/collision/evaluation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from .cage_graph import RouteGraph, RouteCandidate
from .config import AppConfig
from .geometry import Point, Pose, angle_to, distance, wrap_angle
from .lidar_assessment import LidarAssessment, assess_lidar
from .localization import PoseEstimator
from .mapping import FrontierCluster, OccupancyGrid
from .planner import GridPlanner
from .sensors import LidarScan, LidarSensor
from .world import World

@dataclass
class TargetReport:
    detected: bool = False; xy: Point | None = None; confidence: float = 0.0
    source_robot: int = -1; time_s: float = 0.0; reported_home: bool = False

@dataclass
class RobotPacket:
    sender_id: int; time_s: float; map_digest: dict; graph_digest: dict; target_report: dict | None
    task: str; current_goal: Point | None; current_path_digest: list[Point]; visited_digest: list[Point]
    # Full downsampled estimated trajectory from HOME to current position.
    # This is persistent team-history knowledge, unlike the short current path/visit digest.
    trajectory_digest: list[Point]
    estimated_pose: tuple[float, float, float]; pose_cov_trace: float
    assigned_region_id: tuple[int, int] | None = None
    assigned_region_center: Point | None = None
    assigned_region_radius: float = 0.0
    assigned_region_score: float = 0.0

@dataclass
class RobotStatus:
    task: str = "INIT"; planning_source: str = "knowledge map + EKF pose estimate"
    note: str = ""; goal: Point | None = None; last_plan_success: bool = False; last_plan_reason: str = ""
    last_path_min_clearance: float = 0.0; reward_breakdown: dict[str, float] = field(default_factory=dict)

@dataclass
class CoarseRegion:
    region_id: tuple[int, int]
    center: Point
    radius: float
    unknown_cells: int
    known_free_cells: int
    frontier_support: int
    score: float

class RobotAgent:
    def __init__(self, robot_id: int, initial_pose: Pose, cfg: AppConfig, world: World, rng: np.random.Generator):
        self.id=robot_id; self.cfg=cfg; self.rng=rng
        self.true_pose=np.array(initial_pose,dtype=float); self.true_path=[(float(initial_pose[0]),float(initial_pose[1]))]
        self.estimator=PoseEstimator(initial_pose,cfg.motion,rng); self.lidar=LidarSensor(cfg.lidar,rng)
        # self_map contains only this robot's own LiDAR observations.
        # knowledge_map contains everything this robot knows: self_map plus
        # teammate/relay map digests received through communication.  Existing
        # planner/UI code uses robot.map, so keep it as the knowledge map.
        self.self_map=OccupancyGrid(world.width,world.height,cfg.mapping)
        self.knowledge_map=OccupancyGrid(world.width,world.height,cfg.mapping)
        self.map=self.knowledge_map
        self.graph=RouteGraph(cfg.cage.edge_merge_distance)
        self.home_node=self.graph.add_node(world.home,kind="home",confidence=1.0,allow_merge=False)
        self.home_xy=world.home
        self.search_prior_xy=self._sector_prior_point(world)
        self.last_graph_node=self.home_node; self.last_keypoint_xy=world.home
        self.scan: LidarScan | None=None; self.assessment=LidarAssessment(); self.planner=GridPlanner(cfg.planning)
        self.path=[]; self.path_index=0; self.last_replan_time=-999.0; self.current_goal=None; self.current_task="SEARCH"
        self.goal_commit_start=-999.0; self.goal_commit_score=-math.inf; self.best_goal_distance=math.inf; self.last_goal_progress_time=0.0
        self.status=RobotStatus(task="SEARCH"); self.target=TargetReport()
        self.force_return_home=False; self.last_scan_match_time=-999.0
        self.known_teammate_goals={}; self.known_teammate_paths={}; self.known_teammate_visits={}; self.known_teammate_tasks={}
        self.known_teammate_pose={}; self.known_teammate_cov={}; self.known_teammate_last_seen={}
        # Persistent full-from-HOME teammate trajectory memory.  These paths are
        # downsampled estimated trajectories and are not deleted with short-term
        # intent expiry; they describe where this robot believes teammates have been.
        self.known_teammate_trajectories={}; self.known_teammate_trajectory_time={}
        # LOS-realistic coarse exploration intent. A robot only knows teammate
        # regions after receiving packets through the existing LOS communication.
        self.assigned_region: CoarseRegion | None = None
        self.assigned_region_start_time: float = -999.0
        self.known_teammate_regions: dict[int, dict] = {}
        self.visit_history=[self.est_xy]
        self.trajectory_from_home=[self.est_xy]
        self.failed_goal_memory=[]
        # Target-roundtrip state.  Once a target report is known, every robot
        # tries to reach the target from its own current location, records the
        # route attempt, then returns HOME and uploads route/map evidence.
        self.target_reached=False
        self.completed_target_roundtrip=False
        self.target_route_trace=[]
        self.return_route_trace=[]
        self.target_reached_time=-999.0
        self.completed_roundtrip_time=-999.0
        self.route_candidate_uploaded=False
        self.last_command=(0.0,0.0); self.last_pose_quality=1.0; self.best_routes=[]; self.received_packets=0; self.blocked_events=0; self.last_home_full_upload_time=-999.0

    @property
    def est_pose(self)->Pose: return self.estimator.belief.as_pose()
    @property
    def est_xy(self)->Point: return self.estimator.belief.xy
    @property
    def cov_trace(self)->float: return self.estimator.belief.cov_trace_xy

    def _sector_prior_point(self, world: World)->Point:
        hx,hy=world.home
        far=(world.width-world.cfg.world_margin, world.height-world.cfg.world_margin)
        base=angle_to(world.home,far)
        n=max(1,self.cfg.robot.count); spread=math.radians(78.0)
        offset=0.0 if n==1 else (self.id/(n-1)-0.5)*spread
        ang=base+offset; dx,dy=math.cos(ang),math.sin(ang)
        margin=max(0.8,world.cfg.world_margin*0.5); ts=[]
        if dx>1e-6: ts.append((world.width-margin-hx)/dx)
        elif dx<-1e-6: ts.append((margin-hx)/dx)
        if dy>1e-6: ts.append((world.height-margin-hy)/dy)
        elif dy<-1e-6: ts.append((margin-hy)/dy)
        t=max(1.0,min(v for v in ts if v>0)) if any(v>0 for v in ts) else max(world.width,world.height)*0.6
        return (float(min(world.width-margin,max(margin,hx+dx*t))),float(min(world.height-margin,max(margin,hy+dy*t))))

    def step_predict_and_move(self, world: World, dt: float, peer_poses: list[Pose] | None = None) -> None:
        v,omega=self.last_command; x,y,th=self.true_pose
        new_th=wrap_angle(th+omega*dt); cand=(float(x+math.cos(new_th)*v*dt), float(y+math.sin(new_th)*v*dt))
        executed_v, executed_omega = v, omega
        collision_free=self._peer_collision_free(cand,peer_poses or [])
        if world.is_free(cand, margin=self.cfg.robot.radius) and collision_free: self.true_pose[:]=[cand[0],cand[1],new_th]
        else:
            self.blocked_events+=1; self.path=[]; self.status.note="true_robot_collision_prevented_by_sim" if not collision_free else "true_collision_prevented_by_sim"; self.last_command=(0.0,0.0)
            executed_v, executed_omega = 0.0, 0.0
        self._append_true_path_sample()
        self.estimator.predict_from_command(executed_v,executed_omega,dt)

    def _peer_collision_free(self,cand:Point,peer_poses:list[Pose])->bool:
        min_sep=2.0*float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m)
        return all(distance(cand,(float(p[0]),float(p[1])))>=min_sep for p in peer_poses)

    def _append_true_path_sample(self)->None:
        xy=(float(self.true_pose[0]),float(self.true_pose[1]))
        if self.true_path and distance(xy,self.true_path[-1])<self.cfg.robot.true_path_spacing_m:
            return
        self.true_path.append(xy)
        if len(self.true_path)>self.cfg.robot.max_true_path_points:
            del self.true_path[:-self.cfg.robot.max_true_path_points]

    def sense_update_map_and_belief(self, world: World, time_s: float) -> None:
        # Landmarks are sensed from hidden truth, but the estimator receives only
        # noisy range/bearing measurements to known beacon locations.
        visible_landmarks = world.visible_landmarks(tuple(self.true_pose), self.cfg.world.landmark_detection_range)
        self.estimator.update_with_landmarks(
            visible_landmarks,
            self.cfg.world.landmark_detection_range,
            sensor_pose=tuple(self.true_pose),
        )
        self.scan = self.lidar.sense(world, tuple(self.true_pose))
        self._maybe_apply_lidar_scan_match(time_s)

        # Assess scan-map agreement on the map BEFORE inserting this scan; this
        # avoids falsely high confidence from scoring a scan against itself.
        prev = None if self.assessment.decision_note == "init" else self.assessment.consistency
        pre_assessment = assess_lidar(self.self_map, self.est_pose, self.scan, self.cfg.lidar, self.cfg.assessment, prev)
        self.last_pose_quality = self.estimator.quality(
            scan_consistency=pre_assessment.consistency,
            landmark_count=len(visible_landmarks),
        )
        self.self_map.update_from_lidar(self.est_pose, self.scan, self.last_pose_quality, self.id, time_s)
        self.knowledge_map.update_from_lidar(self.est_pose, self.scan, self.last_pose_quality, self.id, time_s)
        # Navigation/planning uses the updated knowledge map, but pose-quality
        # scoring above used the self map to avoid scoring against just-received
        # teammate cells.
        self.assessment = assess_lidar(self.knowledge_map, self.est_pose, self.scan, self.cfg.lidar, self.cfg.assessment, pre_assessment.consistency)
        self._update_visit_history()
        self._detect_target(world, time_s)
        self._update_route_graph(time_s)

    def update_localization_from_teammate(self, teammate:"RobotAgent", world:World, time_s:float)->bool:
        if teammate.id==self.id:
            return False
        my_true=(float(self.true_pose[0]),float(self.true_pose[1]))
        other_true=(float(teammate.true_pose[0]),float(teammate.true_pose[1]))
        true_range=distance(my_true,other_true)
        if true_range>float(self.cfg.motion.teammate_localization_range_m):
            return False
        if not world.segment_free(my_true,other_true,margin=min(0.05,float(self.cfg.robot.radius)*0.25)):
            return False
        bearing=wrap_angle(angle_to(my_true,other_true)-float(self.true_pose[2]))
        z_range=max(0.02,true_range+self.rng.normal(0.0,float(self.cfg.motion.teammate_range_std_m)+0.012*true_range))
        z_bearing=wrap_angle(bearing+self.rng.normal(0.0,math.radians(float(self.cfg.motion.teammate_bearing_std_deg))))
        return self.estimator.update_with_teammate_pose(teammate.est_pose,teammate.cov_trace,z_range,z_bearing)

    def _maybe_apply_lidar_scan_match(self, time_s: float) -> None:
        if self.scan is None:
            return
        if time_s - self.last_scan_match_time < self.cfg.motion.lidar_match_period_s:
            return
        self.last_scan_match_time = time_s
        if np.count_nonzero(self.self_map.quality > 0.05) < 35:
            return

        stride = max(4, len(self.scan.angles) // 14)
        angles = self.scan.angles[::stride]
        ranges = self.scan.ranges[::stride]
        hits = self.scan.hit[::stride]
        if len(angles) < 8:
            return

        base_pose = self.est_pose
        max_xy = float(self.cfg.motion.lidar_match_max_xy_m)
        max_th = math.radians(float(self.cfg.motion.lidar_match_max_theta_deg))
        th = base_pose[2]
        forward = np.array([math.cos(th), math.sin(th)], dtype=float)
        lateral = np.array([-math.sin(th), math.cos(th)], dtype=float)

        def scan_error(pose: Pose, regularization: float = 0.0) -> float:
            pred = self.self_map.predict_scan_ranges(pose, angles, self.cfg.lidar.range)
            active = hits | (pred < self.cfg.lidar.range * 0.96)
            if not np.any(active):
                active = np.ones_like(pred, dtype=bool)
            err = np.abs(pred[active] - ranges[active])
            if len(err) == 0:
                return float("inf")
            # Robust metric: median handles single-ray noise, mean catches broad mismatch.
            return float(np.median(err) + 0.30 * np.mean(err) + regularization)

        base_err = scan_error(base_pose)
        best = (base_err, 0.0, 0.0, 0.0)
        lin_steps = (-1.0, -0.5, 0.0, 0.5, 1.0)
        th_steps = (-1.0, 0.0, 1.0)
        for fs in lin_steps:
            for ls in lin_steps:
                delta = forward * (fs * max_xy) + lateral * (ls * max_xy)
                xy_reg = 0.05 * math.hypot(fs, ls)
                for ts in th_steps:
                    dth = ts * max_th
                    pose = (
                        base_pose[0] + float(delta[0]),
                        base_pose[1] + float(delta[1]),
                        wrap_angle(base_pose[2] + dth),
                    )
                    e = scan_error(pose, regularization=xy_reg + 0.035 * abs(ts))
                    if e < best[0]:
                        best = (e, float(delta[0]), float(delta[1]), float(dth))

        improvement = base_err - best[0]
        # Require a meaningful improvement; otherwise the scan matcher could
        # chase noise or reinforce a drifted self-map.
        if improvement > 0.030:
            confidence = float(np.clip(improvement / 0.42, 0.0, 1.0)) * self.estimator.quality()
            self.estimator.apply_lidar_correction(best[1], best[2], best[3], confidence)

    def _update_visit_history(self)->None:
        xy=self.est_xy
        if not self.visit_history or distance(xy,self.visit_history[-1])>=self.cfg.robot.visit_history_spacing_m:
            self.visit_history.append(xy); self.visit_history=self.visit_history[-self.cfg.robot.max_visit_history:]
        if not self.trajectory_from_home or distance(xy,self.trajectory_from_home[-1])>=self.cfg.robot.trajectory_history_spacing_m:
            self.trajectory_from_home.append((float(xy[0]),float(xy[1])))
            if len(self.trajectory_from_home)>self.cfg.robot.max_trajectory_history_points:
                # Preserve the initial HOME-side point and keep the recent history.
                keep=max(2,self.cfg.robot.max_trajectory_history_points)
                self.trajectory_from_home=[self.trajectory_from_home[0]]+self.trajectory_from_home[-(keep-1):]
        self._append_roundtrip_trace(xy)

    def _activate_target_guidance(self,time_s:float)->None:
        # Start a route attempt only once.  A later, higher-confidence target
        # report may update self.target.xy, but the route should still record
        # the robot's path from when target knowledge first became available.
        if not self.target_route_trace and not self.target_reached:
            self.target_route_trace=[(float(self.est_xy[0]),float(self.est_xy[1]))]

    def _append_roundtrip_trace(self,xy:Point)->None:
        if not self.target.detected or self.completed_target_roundtrip:
            return
        trace = self.return_route_trace if self.target_reached else self.target_route_trace
        if not trace or distance(xy,trace[-1])>=self.cfg.robot.visit_history_spacing_m:
            trace.append((float(xy[0]),float(xy[1])))
            max_len=max(80,self.cfg.robot.max_true_path_points)
            if len(trace)>max_len:
                del trace[:-max_len]

    def mark_target_reached(self,time_s:float)->None:
        if self.target_reached:
            return
        self.target_reached=True
        self.target_reached_time=float(time_s)
        self.current_goal=None
        self.path=[]
        self.path_index=0
        if not self.target_route_trace:
            self.target_route_trace=[(float(self.est_xy[0]),float(self.est_xy[1]))]
        self.target_route_trace.append((float(self.est_xy[0]),float(self.est_xy[1])))
        self.return_route_trace=[(float(self.est_xy[0]),float(self.est_xy[1]))]
        self.status.note=f"target_reached_by_R{self.id}_returning_home"

    def mark_target_roundtrip_complete(self,time_s:float)->None:
        if self.completed_target_roundtrip:
            return
        self.completed_target_roundtrip=True
        self.completed_roundtrip_time=float(time_s)
        self.current_goal=None
        self.path=[]
        self.path_index=0
        if self.return_route_trace:
            self.return_route_trace.append((float(self.est_xy[0]),float(self.est_xy[1])))
        self.status.note=f"target_roundtrip_complete_R{self.id}"

    def target_route_summary(self)->dict|None:
        if not self.target_reached or len(self.target_route_trace)<2:
            return None
        path=[(float(x),float(y)) for x,y in self.target_route_trace]
        ret=[(float(x),float(y)) for x,y in self.return_route_trace]
        def path_len(pts:list[Point])->float:
            return float(sum(distance(a,b) for a,b in zip(pts[:-1],pts[1:]))) if len(pts)>=2 else 0.0
        cells=[]; seen=set()
        for pts in (path,ret):
            for a,b in zip(pts[:-1],pts[1:]):
                ca=self.self_map.world_to_cell(a); cb=self.self_map.world_to_cell(b)
                if ca is None or cb is None: continue
                for c in self.self_map._bresenham(ca,cb):
                    if c not in seen:
                        seen.add(c); cells.append(c)
        known=self.self_map.known_mask()
        clearance=self.self_map.clearance_map(max_radius_m=max(3.0,self.cfg.planning.desired_clearance_m*3.0))
        q_vals=[]; cl_vals=[]; unknown=0
        for i,j in cells:
            if not known[j,i]: unknown+=1
            q_vals.append(float(self.self_map.quality[j,i]))
            cl_vals.append(float(clearance[j,i]))
        total=max(1,len(cells))
        return {
            "robot_id":int(self.id),
            "target_reached":bool(self.target_reached),
            "roundtrip_complete":bool(self.completed_target_roundtrip),
            "target_reached_time":float(self.target_reached_time),
            "completed_time":float(self.completed_roundtrip_time),
            "target_xy":[float(self.target.xy[0]),float(self.target.xy[1])] if self.target.xy else None,
            "route_to_target":path,
            "return_route":ret,
            "route_length":path_len(path),
            "return_length":path_len(ret),
            "mean_quality":float(np.mean(q_vals)) if q_vals else 0.0,
            "min_clearance":float(min(cl_vals)) if cl_vals else 0.0,
            "unknown_fraction":float(unknown)/float(total),
        }

    def _detect_target(self, world: World, time_s: float)->None:
        if self.target.detected or not world.target_visible(tuple(self.true_pose), self.cfg.lidar.range): return
        true_xy=(float(self.true_pose[0]),float(self.true_pose[1])); brg=angle_to(true_xy,world.target)-float(self.true_pose[2]); rr=distance(true_xy,world.target)
        r=max(0.05,rr+self.rng.normal(0.0,0.06)); b=brg+self.rng.normal(0.0,math.radians(2.0))
        ex,ey,eth=self.est_pose; est_target=(float(ex+math.cos(eth+b)*r), float(ey+math.sin(eth+b)*r))
        conf=float(np.clip(self.assessment.consistency*self.last_pose_quality,0.1,1.0))
        self.target=TargetReport(True,est_target,conf,self.id,time_s,False)
        self._activate_target_guidance(time_s)
        tid=self.graph.add_node(est_target,kind="target",confidence=conf,allow_merge=True); self.graph.target_id=tid
        clearance=max(0.05,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        self.graph.add_or_update_edge(self.last_graph_node,tid,clearance=clearance,consistency=max(0.05,self.assessment.consistency),pose_quality=self.last_pose_quality,robot_id=self.id,time_s=time_s,success=True)
        self.status.note=f"target_detected_by_R{self.id}"

    def _update_route_graph(self,time_s:float)->None:
        xy=self.est_xy
        if distance(xy,self.last_keypoint_xy)<self.cfg.robot.keypoint_spacing: return
        kind="anchor" if self.assessment.consistency>0.68 and self.last_pose_quality>0.45 else "keypoint"
        node=self.graph.add_node(xy,kind=kind,confidence=max(self.assessment.consistency,self.last_pose_quality))
        clearance=max(0.05,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        self.graph.add_or_update_edge(self.last_graph_node,node,clearance=clearance,consistency=max(0.05,self.assessment.consistency),pose_quality=self.last_pose_quality,robot_id=self.id,time_s=time_s,success=not self.assessment.blocked_forward)
        self.last_graph_node=node; self.last_keypoint_xy=xy

    def receive_packet(self, packet: RobotPacket)->None:
        if packet.sender_id==self.id: return
        self.received_packets+=1
        # Robot-to-robot packets can carry communication-limited knowledge maps.
        # Merge them into knowledge_map only; self_map remains pure own LiDAR.
        if packet.sender_id >= 0 and packet.map_digest:
            self.knowledge_map.merge_from_digest(packet.map_digest, combine_sources=True)
        self.graph.merge_from_digest(packet.graph_digest)
        if packet.sender_id>=0:
            self.known_teammate_pose[packet.sender_id]=packet.estimated_pose; self.known_teammate_cov[packet.sender_id]=float(packet.pose_cov_trace)
            self.known_teammate_last_seen[packet.sender_id]=float(packet.time_s); self.known_teammate_tasks[packet.sender_id]=packet.task
            if packet.current_goal is not None: self.known_teammate_goals[packet.sender_id]=(float(packet.current_goal[0]),float(packet.current_goal[1]))
            else: self.known_teammate_goals.pop(packet.sender_id,None)
            self.known_teammate_paths[packet.sender_id]=[(float(x),float(y)) for x,y in packet.current_path_digest]
            self.known_teammate_visits[packet.sender_id]=[(float(x),float(y)) for x,y in packet.visited_digest]
            if packet.trajectory_digest:
                self.known_teammate_trajectories[packet.sender_id]=[(float(x),float(y)) for x,y in packet.trajectory_digest]
                self.known_teammate_trajectory_time[packet.sender_id]=float(packet.time_s)
            if packet.assigned_region_center is not None and packet.assigned_region_id is not None:
                cx,cy=packet.assigned_region_center
                self.known_teammate_regions[packet.sender_id]={
                    "region_id": tuple(packet.assigned_region_id),
                    "center": (float(cx),float(cy)),
                    "radius": float(packet.assigned_region_radius),
                    "score": float(packet.assigned_region_score),
                    "time_s": float(packet.time_s),
                    "task": str(packet.task),
                }
            else:
                self.known_teammate_regions.pop(packet.sender_id,None)
        target_share_allowed = packet.sender_id == -1 or bool(self.cfg.target_reporting.allow_robot_to_robot_target_share)
        if target_share_allowed and packet.target_report and packet.target_report.get("detected"):
            tr=packet.target_report; conf=float(tr.get("confidence",0.0))
            if not self.target.detected or conf>self.target.confidence:
                xy=tuple(tr["xy"]); self.target=TargetReport(True,(float(xy[0]),float(xy[1])),conf,int(tr.get("source_robot",packet.sender_id)),float(tr.get("time_s",packet.time_s)),bool(tr.get("reported_home",False)))
                self._activate_target_guidance(float(tr.get("time_s",packet.time_s)))
                tid=self.graph.add_node(self.target.xy,kind="target",confidence=conf,allow_merge=True); self.graph.target_id=tid
            elif bool(tr.get("reported_home",False)):
                self.target.reported_home=True
        self._expire_stale_teammate_intent(packet.time_s)

    def make_packet(self,time_s:float,include_map_digest:bool=True,max_map_cells:int|None=650,map_source:str="knowledge")->RobotPacket:
        """Create a packet. Robot-to-robot packets use knowledge_map; HOME uploads use self_map."""
        target_dict=None
        if self.target.detected and self.target.xy is not None:
            target_dict={"detected":True,"xy":[float(self.target.xy[0]),float(self.target.xy[1])],"confidence":float(self.target.confidence),"source_robot":int(self.target.source_robot),"time_s":float(self.target.time_s),"reported_home":bool(self.target.reported_home)}
        map_digest={}
        if include_map_digest:
            if map_source=="self":
                map_obj=self.self_map
            elif map_source=="knowledge":
                map_obj=self.knowledge_map
            else:
                raise ValueError(f"unknown map_source {map_source!r}")
            map_digest=map_obj.make_digest(self.id,time_s,max_cells=max_map_cells)
        region_id=None; region_center=None; region_radius=0.0; region_score=0.0
        if self.assigned_region is not None and self.current_task in {"SEARCH_HIER_NBV","SEARCH_NBV","DEPLOY_FROM_HOME"}:
            region_id=tuple(self.assigned_region.region_id)
            region_center=(float(self.assigned_region.center[0]),float(self.assigned_region.center[1]))
            region_radius=float(self.assigned_region.radius)
            region_score=float(self.assigned_region.score)
        return RobotPacket(self.id,float(time_s),map_digest,self.graph.make_digest(self.id,time_s),target_dict,self.current_task,self.current_goal,self._path_digest(),self._visited_digest(),self._trajectory_digest(),self.est_pose,self.cov_trace,region_id,region_center,region_radius,region_score)

    def make_full_knowledge_packet(self,time_s:float)->RobotPacket:
        return self.make_packet(time_s,include_map_digest=True,max_map_cells=None,map_source="knowledge")

    def make_full_self_packet(self,time_s:float)->RobotPacket:
        return self.make_packet(time_s,include_map_digest=True,max_map_cells=None,map_source="self")

    def make_partial_self_packet(self,time_s:float,max_map_cells:int|None=650)->RobotPacket:
        return self.make_packet(time_s,include_map_digest=True,max_map_cells=max_map_cells,map_source="self")

    def _path_digest(self)->list[Point]:
        if not self.path or self.path_index>=len(self.path): return []
        pts=[self.est_xy]+self.path[self.path_index:]; out=[]; last=None
        for p in pts:
            if last is None or distance(last,p)>=self.cfg.robot.path_digest_spacing_m:
                out.append((float(p[0]),float(p[1]))); last=p
            if len(out)>=self.cfg.robot.max_path_digest_points: break
        return out

    def _visited_digest(self)->list[Point]:
        out=[]; last=None
        for p in reversed(self.visit_history):
            if last is None or distance(last,p)>=self.cfg.robot.visit_digest_spacing_m:
                out.append((float(p[0]),float(p[1]))); last=p
            if len(out)>=self.cfg.robot.max_visit_digest_points: break
        out.reverse()
        return out

    def _trajectory_digest(self)->list[Point]:
        """Full downsampled estimated path from HOME to current position."""
        pts=list(self.trajectory_from_home)
        if not pts:
            return []
        if distance(pts[-1],self.est_xy)>0.15:
            pts.append(self.est_xy)
        spacing=max(0.05,float(self.cfg.robot.trajectory_digest_spacing_m))
        out=[]; last=None
        for p in pts:
            if last is None or distance(last,p)>=spacing:
                out.append((float(p[0]),float(p[1]))); last=p
        # Always keep the current endpoint.
        end=(float(pts[-1][0]),float(pts[-1][1]))
        if not out or distance(out[-1],end)>0.15:
            out.append(end)
        max_pts=max(2,int(self.cfg.robot.max_trajectory_digest_points))
        if len(out)>max_pts:
            idx=np.linspace(0,len(out)-1,max_pts).astype(int)
            out=[out[int(i)] for i in idx]
        return out

    def choose_task_and_plan(self,time_s:float,reserved_goals:dict[int,Point]|None=None,reserved_frontiers:dict[int,Point]|None=None)->None:
        self._expire_stale_teammate_intent(time_s); team_goals=self.fresh_teammate_goals(time_s); team_paths=self.fresh_teammate_paths(time_s); team_visits=self.fresh_teammate_visits(time_s); team_trajectories=self.known_teammate_trajectories_snapshot()
        dynamic_obstacles=self._teammate_dynamic_obstacles()
        if reserved_goals:
            for rid,g in reserved_goals.items():
                if rid!=self.id and g is not None:
                    team_goals[-1000-rid]=(float(g[0]),float(g[1]))
        event_replan=False
        if self.assessment.blocked_forward and self.path:
            self._remember_failed_goal(); self.path=[]; self.path_index=0; self.status.note="path_invalidated_by_lidar_block"; event_replan=True
        if self._goal_progress_stalled(time_s):
            self._remember_failed_goal(); self.path=[]; self.path_index=0; self.status.note="goal_progress_stalled"; event_replan=True
        if self._should_keep_committed_goal(time_s,event_replan): return
        target_mode = self.target.detected and not self.completed_target_roundtrip
        force_target_replan = target_mode and self.current_task not in {"GO_TO_TARGET","EXPLORE_TOWARD_TARGET","RETURN_HOME_AFTER_TARGET","WAIT_AT_HOME_DONE"}
        if not event_replan and not force_target_replan and time_s-self.last_replan_time<self.cfg.robot.path_replan_period_s:
            goal_reached=self.current_goal is not None and distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_tolerance
            if self.path_index<len(self.path) or not goal_reached:
                return
        goal,task,reason,breakdown=self._select_goal_from_lidar_map(team_goals,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_frontiers or {},time_s)
        if self._should_reject_goal_switch(goal,breakdown,time_s,event_replan):
            self.status.note="committed_current_goal"
            self.last_replan_time=time_s
            return
        self.current_goal=goal; self.current_task=task; self.status.task=task; self.status.goal=goal
        self.status.planning_source="communication-limited knowledge map + EKF pose estimate"; self.status.note=reason; self.status.reward_breakdown=breakdown
        if goal is None:
            self.path=[]; self.path_index=0; self.status.last_plan_success=False; self.status.last_plan_reason="no_goal_available"; self.status.last_path_min_clearance=0.0; self.last_replan_time=time_s; return
        result=self.planner.plan(self.map,self.est_xy,goal,dynamic_obstacles=dynamic_obstacles)
        if result.success and len(result.path)>=2:
            simplified=self._downsample_path(result.path,spacing=0.45); simp_clear=self.map.path_min_clearance(simplified)
            self.path=simplified if simp_clear>=self.cfg.planning.critical_clearance_m else result.path; self.path_index=0
            self.status.last_path_min_clearance=max(0.0,min(result.min_clearance,simp_clear if simplified else result.min_clearance))
        elif task in {"REPORT_TARGET_HOME","RETURN_HOME_CERT_ROUTE","RETURN_HOME_AFTER_TARGET","RETURN_HOME_EXPLORATION_COMPLETE"}:
            self.path=self._homing_fallback_path(goal); self.path_index=0
            result.success=bool(self.path); result.reason="homing_fallback" if self.path else result.reason
            self.status.last_path_min_clearance=max(0.0,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        elif task in {"GO_TO_TARGET","EXPLORE_TOWARD_TARGET"}:
            self.path=self._target_fallback_path(goal); self.path_index=0
            result.success=bool(self.path); result.reason="target_directed_fallback" if self.path else result.reason
            self.status.last_path_min_clearance=max(0.0,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        else:
            self._remember_failed_goal(goal); self.path=[]; self.path_index=0; self.status.last_path_min_clearance=0.0
        self.status.last_plan_success=result.success; self.status.last_plan_reason=result.reason; self.last_replan_time=time_s; self.best_routes=self.graph.top_routes(k=4)
        if result.success and self.path:
            self.current_goal=self.path[-1]; self.status.goal=self.current_goal
        if result.success:
            self.goal_commit_start=time_s; self.goal_commit_score=float(breakdown.get("score",0.0)); self.best_goal_distance=distance(self.est_xy,self.current_goal if self.current_goal is not None else goal); self.last_goal_progress_time=time_s

    def _remember_failed_goal(self, goal: Point | None = None)->None:
        g=goal if goal is not None else self.current_goal
        if g is None: return
        self.failed_goal_memory.append((float(g[0]),float(g[1])))
        self.failed_goal_memory=self.failed_goal_memory[-self.cfg.robot.failed_goal_memory_size:]

    def _goal_progress_stalled(self,time_s:float)->bool:
        if self.current_goal is None or not self.path or self.path_index>=len(self.path): return False
        d=distance(self.est_xy,self.current_goal)
        if d+0.35<self.best_goal_distance:
            self.best_goal_distance=d; self.last_goal_progress_time=time_s
            return False
        return time_s-self.last_goal_progress_time>self.cfg.robot.stuck_progress_timeout_s

    def _should_keep_committed_goal(self,time_s:float,event_replan:bool)->bool:
        if event_replan or self.current_goal is None or not self.path or self.path_index>=len(self.path): return False
        if self.target.detected: return False
        current_d=distance(self.est_xy,self.current_goal)
        if current_d<=self.cfg.robot.goal_tolerance: return False
        if time_s-self.goal_commit_start<self.cfg.robot.goal_commit_time_s:
            self.status.note="commit_hold"
            return True
        if current_d<=self.cfg.robot.goal_finish_commit_radius_m:
            self.status.note="commit_finish_current_goal"
            return True
        return False

    def _should_reject_goal_switch(self,goal:Point|None,breakdown:dict[str,float],time_s:float,event_replan:bool)->bool:
        if event_replan or goal is None or self.current_goal is None or not self.path or self.path_index>=len(self.path): return False
        if self.target.detected or distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_tolerance: return False
        if distance(goal,self.current_goal)<self.cfg.robot.goal_switch_same_goal_radius_m: return False
        new_score=float(breakdown.get("score",0.0))
        required_gain=float(self.cfg.robot.goal_switch_score_margin)
        if time_s-self.last_goal_progress_time<=self.cfg.robot.stuck_progress_timeout_s:
            required_gain+=float(self.cfg.robot.goal_progress_switch_margin)
        if distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_finish_commit_radius_m*1.35:
            required_gain+=float(self.cfg.robot.goal_finish_switch_margin)
        return new_score<self.goal_commit_score+required_gain

    def fresh_teammate_goals(self,time_s:float)->dict[int,Point]: self._expire_stale_teammate_intent(time_s); return dict(self.known_teammate_goals)
    def fresh_teammate_paths(self,time_s:float)->dict[int,list[Point]]: self._expire_stale_teammate_intent(time_s); return {rid:list(path) for rid,path in self.known_teammate_paths.items() if path}
    def fresh_teammate_visits(self,time_s:float)->dict[int,list[Point]]: self._expire_stale_teammate_intent(time_s); return {rid:list(path) for rid,path in self.known_teammate_visits.items() if path}
    def known_teammate_trajectories_snapshot(self)->dict[int,list[Point]]:
        return {rid:list(path) for rid,path in self.known_teammate_trajectories.items() if path}
    def _teammate_dynamic_obstacles(self)->list[tuple[Point,float]]:
        out:list[tuple[Point,float]]=[]
        base=2.0*float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m)
        max_extra=float(self.cfg.planning.dynamic_obstacle_max_cov_extra_m)
        for rid,pose in self.known_teammate_pose.items():
            if rid==self.id:
                continue
            cov=float(self.known_teammate_cov.get(rid,0.0))
            cov_extra=min(max_extra,0.65*math.sqrt(max(0.0,cov)*0.5))
            out.append(((float(pose[0]),float(pose[1])),base+cov_extra))
        path_radius=float(self.cfg.robot.radius)+0.5*float(self.cfg.robot.collision_buffer_m)
        for rid,path in self.known_teammate_paths.items():
            if rid==self.id or not path:
                continue
            stride=max(1,len(path)//8)
            for p in path[::stride][:10]:
                out.append(((float(p[0]),float(p[1])),path_radius))
        return out[:48]
    def _expire_stale_teammate_intent(self,time_s:float)->None:
        stale=[rid for rid,stamp in self.known_teammate_last_seen.items() if time_s-stamp>self.cfg.communication.teammate_intent_timeout_s]
        for rid in stale:
            for d in (self.known_teammate_last_seen,self.known_teammate_goals,self.known_teammate_paths,self.known_teammate_visits,self.known_teammate_tasks,self.known_teammate_pose,self.known_teammate_cov,self.known_teammate_regions): d.pop(rid,None)

    def _select_goal_from_lidar_map(self,team_goals:dict[int,Point],team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]],dynamic_obstacles:list[tuple[Point,float]],reserved_frontiers:dict[int,Point]|None=None,time_s:float=0.0)->tuple[Point|None,str,str,dict[str,float]]:
        """Choose the next task/goal from communication-limited knowledge.

        Normal exploration was intentionally simplified in this version:
        find reachable frontiers, prefer ones that expose unknown cells, and
        avoid regions already covered by teammate paths.  Target and return-home
        workflows remain separate so exploration tuning does not pollute the
        target roundtrip behavior.
        """
        reserved_frontiers=reserved_frontiers or {}
        target_known = self.target.detected and self.target.xy is not None
        if self.completed_target_roundtrip:
            return None,"WAIT_AT_HOME_DONE","target_roundtrip_complete_wait_at_home",{"roundtrip_complete":1.0}
        if self.target_reached:
            home_dist=distance(self.est_xy,self.home_xy)
            if home_dist>self.cfg.robot.goal_tolerance:
                return self.home_xy,"RETURN_HOME_AFTER_TARGET","target_reached_return_home",{"distance_home":home_dist}
            return None,"WAIT_AT_HOME_DONE","target_roundtrip_complete_wait_at_home",{"distance_home":home_dist}
        if self.force_return_home and not target_known:
            home_dist=distance(self.est_xy,self.home_xy)
            if home_dist>self.cfg.robot.goal_tolerance:
                return self.home_xy,"RETURN_HOME_EXPLORATION_COMPLETE","exploration_complete_return_home",{"distance_home":home_dist}
            return None,"WAIT_AT_HOME","exploration_complete_wait_at_home",{"distance_home":home_dist}
        if self.assessment.consistency<self.cfg.cage.reanchor_consistency_threshold:
            anchor=self._nearest_anchor()
            if anchor is not None and distance(anchor,self.est_xy)>0.35:
                return anchor,"REANCHOR","low_scan_map_consistency_reanchor",{"consistency":self.assessment.consistency}

        # Special startup behavior: before the robots have useful history, push
        # each one out from HOME along a different assigned direction.  This
        # prevents all robots from fighting for the same first frontier.
        if (not target_known) and self.cfg.planning.startup_deployment_enabled:
            deploy=self._select_startup_deployment_goal(dynamic_obstacles,reserved_frontiers)
            if deploy is not None:
                goal,bd=deploy
                return goal,"DEPLOY_FROM_HOME","startup_deployment_spread_from_home",bd

        if target_known:
            target_goal=(float(self.target.xy[0]),float(self.target.xy[1]))
            if distance(self.est_xy,target_goal)<=self.cfg.robot.goal_tolerance:
                probe=self._target_probe_goal(target_goal, allow_beyond_target=True)
                if probe is not None:
                    return probe,"EXPLORE_TOWARD_TARGET","estimated_target_close_probe_for_physical_confirmation",{"target_probe":1.0,"target_progress":self._target_progress_reward(probe)}
            direct=self.planner.plan(self.map,self.est_xy,target_goal,dynamic_obstacles=dynamic_obstacles)
            unknown_frac=self._path_unknown_fraction(direct.path) if direct.success else 1.0
            if direct.success and direct.min_clearance>=self.cfg.planning.critical_clearance_m:
                return target_goal,"GO_TO_TARGET","target_known_drive_to_target_map_while_moving",{
                    "target_distance":distance(self.est_xy,target_goal),
                    "target_path_unknown_fraction":float(unknown_frac),
                    "raw_clearance_m":float(direct.min_clearance),
                    "score":25.0+max(0.0,10.0-distance(self.est_xy,target_goal)),
                }

        clearance_map=self.map.clearance_map(max_radius_m=max(3.0,self.cfg.planning.desired_clearance_m*2.5))
        free_mask=self.map.free_mask(); known_mask=self.map.known_mask(); unknown_mask=~known_mask; occupied_mask=self.map.occupied_mask()

        # Target-guided behavior can still use frontiers as hints near the
        # target corridor.  Normal exploration below does not choose frontier
        # cells directly; it chooses scan poses by expected LiDAR information.
        frontiers=self.map.find_frontiers(self.cfg.planning.frontier_min_cluster_size,self.cfg.planning.frontier_info_radius_m)
        if target_known:
            if frontiers:
                best=self._select_target_directed_frontier_goal(frontiers,clearance_map,free_mask,known_mask,unknown_mask,occupied_mask,dynamic_obstacles)
                if best is not None:
                    return best[1],"EXPLORE_TOWARD_TARGET","target_directed_frontier_selected",best[2]
            probe=self._target_probe_goal((float(self.target.xy[0]),float(self.target.xy[1])))
            if probe is not None:
                return probe,"EXPLORE_TOWARD_TARGET","target_directed_probe_after_frontier_filter",{"target_probe":1.0,"target_progress":self._target_progress_reward(probe)}
            return None,"WAIT","target_known_no_reachable_goal",{}

        # Normal exploration: use a TARE-style hierarchy, but keep it LOS-realistic.
        # First choose a coarse unknown region from this robot's communication-limited
        # knowledge map, then run the existing dense NBV scan-pose selector only
        # around that region.
        best=self._select_hierarchical_nbv_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_frontiers,time_s)
        if best is not None:
            return best[1],"SEARCH_HIER_NBV","hierarchical_region_guided_nbv_selected",best[2]

        # Last fallback: if NBV cannot find a reachable scan pose, move through
        # the best open sector to create new observations.
        probe=self._sector_probe_goal()
        if probe is not None:
            return probe,"SEARCH_OPEN_SECTOR","nbv_no_candidate_use_open_sector_probe",{"open_sector":1.0}
        gx=self.est_xy[0]+math.cos(self.est_pose[2]+self.assessment.best_open_angle)*min(1.8,self.cfg.lidar.range*0.38)
        gy=self.est_xy[1]+math.sin(self.est_pose[2]+self.assessment.best_open_angle)*min(1.8,self.cfg.lidar.range*0.38)
        return (gx,gy),"SEARCH_OPEN_SECTOR","nbv_no_candidate_use_best_lidar_open_sector",{"open_sector":1.0}

    def _select_startup_deployment_goal(self,dynamic_obstacles:list[tuple[Point,float]],reserved_frontiers:dict[int,Point]|None=None)->tuple[Point,dict[str,float]]|None:
        done_radius=0.85*float(self.cfg.planning.startup_deployment_lidar_fraction)*float(self.cfg.lidar.range)
        home_d=distance(self.est_xy,self.home_xy)
        if home_d>=done_radius:
            return None
        n=max(1,int(self.cfg.robot.count))
        # Center launch directions on the rough HOME-to-far-corner exploration
        # direction, but spread robots broadly so startup is not chaotic.
        far=(self.map.width_m-float(self.cfg.world.world_margin),self.map.height_m-float(self.cfg.world.world_margin))
        base=angle_to(self.home_xy,far)
        spread=math.radians(float(self.cfg.planning.startup_deployment_angle_spread_deg))
        offset=0.0 if n==1 else (self.id/(n-1)-0.5)*spread
        assigned=base+offset
        deploy_dist=max(float(self.cfg.lidar.range)*float(self.cfg.planning.startup_deployment_lidar_fraction),self.cfg.robot.goal_tolerance*3.0)
        samples=4
        step=math.radians(12.0)
        angle_offsets=[0.0]
        for k in range(1,9):
            angle_offsets.extend([k*step,-k*step])
        clearance=self.map.clearance_map(max_radius_m=max(3.0,self.cfg.planning.desired_clearance_m*2.5))
        known=self.map.known_mask(); unknown=~known; occupied=self.map.occupied_mask()
        best=None
        reserved=reserved_frontiers or {}
        reservation_radius=max(0.2,float(self.cfg.planning.nbv_reservation_lidar_fraction)*float(self.cfg.lidar.range))
        for dist_scale in np.linspace(1.0,0.55,samples):
            dist_m=deploy_dist*float(dist_scale)
            for off in angle_offsets:
                a=assigned+float(off)
                goal=(self.home_xy[0]+math.cos(a)*dist_m,self.home_xy[1]+math.sin(a)*dist_m)
                cell=self.map.world_to_cell(goal)
                if cell is None:
                    continue
                if any(distance(goal,g)<reservation_radius for rid,g in reserved.items() if rid!=self.id and g is not None):
                    continue
                cl=float(clearance[cell[1],cell[0]])
                visibility=self._expected_lidar_visibility_gain(goal,unknown,occupied)
                angle_pen=abs(float(off))/max(step,1e-6)
                dist_from_robot=distance(self.est_xy,goal)
                score=3.0*dist_scale+0.70*math.log1p(visibility)+1.1*min(1.5,cl/max(1e-6,self.cfg.planning.desired_clearance_m))-0.22*angle_pen-0.04*dist_from_robot
                bd={
                    "score":float(score),
                    "deployment_home_distance":float(home_d),
                    "deployment_done_radius_m":float(done_radius),
                    "deployment_goal_distance_m":float(dist_m),
                    "deployment_angle_offset_deg":float(math.degrees(off)),
                    "expected_visibility":float(visibility),
                    "raw_clearance_m":float(cl),
                    "deployment_distance_from_robot_m":float(dist_from_robot),
                }
                if best is None or score>best[0]:
                    best=(float(score),goal,bd)
        if best is None:
            return None
        return best[1],best[2]

    def _select_target_directed_frontier_goal(self,frontiers:list[FrontierCluster],clearance_map:np.ndarray,free_mask:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,dynamic_obstacles:list[tuple[Point,float]])->tuple[float,Point,dict[str,float]]|None:
        candidates=[]
        limit=max(1,min(len(frontiers),int(self.cfg.planning.frontier_sample_count)))
        for fr in frontiers[:limit]:
            approach=self._exploration_safe_approach_point(fr,clearance_map,free_mask,known_mask)
            d=max(0.1,distance(self.est_xy,approach))
            if d < max(0.75,self.cfg.robot.goal_tolerance*1.5):
                continue
            target_progress=self._target_progress_reward(approach)
            robot_target_corridor=self._robot_target_corridor_reward(approach)
            if target_progress < -0.05 and robot_target_corridor < 0.35:
                continue
            cell=self.map.world_to_cell(approach)
            clearance=float(clearance_map[cell[1],cell[0]]) if cell is not None else 0.0
            visibility=self._expected_lidar_visibility_gain(approach,unknown_mask,occupied_mask)
            info=math.log1p(fr.information_gain)+0.8*math.log1p(visibility)
            corridor=self._target_corridor_reward(approach)
            corridor_lowq=self._target_corridor_low_quality_reward(approach)
            score=0.45*info+1.2*min(1.5,clearance/max(1e-6,self.cfg.planning.desired_clearance_m))+8.0*target_progress+5.0*robot_target_corridor+self.cfg.cage.target_corridor_bonus_weight*corridor+self.cfg.cage.target_corridor_low_quality_weight*corridor_lowq-0.65*self.cfg.planning.distance_weight*d
            bd={"score":float(score),"info":float(info),"frontier_gain":float(fr.information_gain),"expected_visibility":float(visibility),"raw_clearance_m":float(clearance),"distance_cost":float(0.65*self.cfg.planning.distance_weight*d),"target_progress":float(target_progress),"robot_target_corridor":float(robot_target_corridor),"target_corridor":float(self.cfg.cage.target_corridor_bonus_weight*corridor),"corridor_lowq":float(self.cfg.cage.target_corridor_low_quality_weight*corridor_lowq),"target_mode_ignores_team_path_penalties":1.0}
            candidates.append((score,approach,bd))
        return self._best_planned_frontier_candidate(candidates,dynamic_obstacles)

    def _select_hierarchical_nbv_goal(self,clearance_map:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]],dynamic_obstacles:list[tuple[Point,float]],reserved_targets:dict[int,Point]|None,time_s:float)->tuple[float,Point,dict[str,float]]|None:
        if not bool(self.cfg.planning.hierarchical_exploration_enabled):
            self.assigned_region=None
            best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=False)
            if best is None:
                best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=True)
            return best

        team_points=self._team_history_points(team_paths,team_visits,team_trajectories)
        hard_radius=float(self.cfg.planning.nbv_teammate_hard_avoid_lidar_fraction)*float(self.cfg.lidar.range)
        soft_radius=max(hard_radius,float(self.cfg.planning.nbv_teammate_soft_avoid_lidar_fraction)*float(self.cfg.lidar.range))
        regions=self._build_coarse_exploration_regions(known_mask,unknown_mask,occupied_mask,clearance_map,team_points,reserved_targets or {},hard_radius,soft_radius)
        ordered=self._ordered_region_choices(regions,time_s)
        tried=0
        for region in ordered[:5]:
            tried+=1
            for relaxed in (False,True):
                best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=relaxed,focus_region=region)
                if best is None:
                    continue
                switched=self.assigned_region is None or self.assigned_region.region_id!=region.region_id
                self.assigned_region=region
                if switched:
                    self.assigned_region_start_time=time_s
                bd=dict(best[2])
                bd.update({
                    "hierarchical_exploration":1.0,
                    "coarse_region_i":float(region.region_id[0]),
                    "coarse_region_j":float(region.region_id[1]),
                    "coarse_region_center_x":float(region.center[0]),
                    "coarse_region_center_y":float(region.center[1]),
                    "coarse_region_radius_m":float(region.radius),
                    "coarse_region_score":float(region.score),
                    "coarse_region_unknown_cells":float(region.unknown_cells),
                    "coarse_region_known_free_cells":float(region.known_free_cells),
                    "coarse_region_frontier_support":float(region.frontier_support),
                    "coarse_regions_available":float(len(regions)),
                    "coarse_regions_tried":float(tried),
                    "coarse_region_relaxed_nbv":float(1.0 if relaxed else 0.0),
                })
                return best[0],best[1],bd

        # Region layer failed to produce a reachable local scan pose.  Fall back to
        # the existing global NBV instead of stopping exploration.
        self.assigned_region=None
        best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=False)
        if best is None:
            best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=True)
        if best is not None:
            bd=dict(best[2]); bd.update({"hierarchical_exploration":0.5,"hierarchical_fallback_global_nbv":1.0,"coarse_regions_available":float(len(regions))})
            return best[0],best[1],bd
        return None

    def _ordered_region_choices(self,regions:list[CoarseRegion],time_s:float)->list[CoarseRegion]:
        if not regions:
            self.assigned_region=None
            return []
        by_id={r.region_id:r for r in regions}
        current=by_id.get(self.assigned_region.region_id) if self.assigned_region is not None else None
        best=max(regions,key=lambda r:r.score)
        ordered:list[CoarseRegion]=[]
        if current is not None:
            commit_time=float(self.cfg.planning.region_commit_time_s)
            switch_ratio=max(1.0,float(self.cfg.planning.region_switch_score_ratio))
            if time_s-self.assigned_region_start_time<commit_time:
                ordered.append(current)
            elif current.score>0.0 and best.score<current.score*switch_ratio:
                ordered.append(current)
        if best not in ordered:
            ordered.append(best)
        for r in sorted(regions,key=lambda r:r.score,reverse=True):
            if r not in ordered:
                ordered.append(r)
            if len(ordered)>=8:
                break
        return ordered

    def _build_coarse_exploration_regions(self,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,clearance_map:np.ndarray,team_points:list[Point],reserved_targets:dict[int,Point],hard_radius:float,soft_radius:float)->list[CoarseRegion]:
        res=float(self.map.res)
        region_size=max(4.0*res,float(self.cfg.lidar.range)*float(self.cfg.planning.region_size_lidar_fraction))
        block=max(3,int(math.ceil(region_size/res)))
        radius=0.5*math.sqrt(2.0)*block*res
        known_free=known_mask & (~occupied_mask)
        min_unknown=max(4,int(0.08*block*block))
        regions:list[CoarseRegion]=[]
        rid_j=0
        for y0 in range(0,self.map.ny,block):
            y1=min(self.map.ny,y0+block)
            rid_i=0
            for x0 in range(0,self.map.nx,block):
                x1=min(self.map.nx,x0+block)
                sub_unknown=unknown_mask[y0:y1,x0:x1]
                unknown_count=int(np.count_nonzero(sub_unknown))
                if unknown_count<min_unknown:
                    rid_i+=1; continue
                sub_occ=occupied_mask[y0:y1,x0:x1]
                if np.count_nonzero(sub_occ)>0.62*sub_occ.size:
                    rid_i+=1; continue
                cx=(x0+x1)*0.5*res; cy=(y0+y1)*0.5*res
                center=(float(cx),float(cy))
                ci=int(min(self.map.nx-1,max(0,round(cx/res-0.5))))
                cj=int(min(self.map.ny-1,max(0,round(cy/res-0.5))))
                support_r=max(2,block//2+2)
                known_free_count=self._local_cell_count(known_free,ci,cj,support_r)
                if known_free_count<2 and distance(center,self.est_xy)>float(self.cfg.lidar.range)*1.10:
                    rid_i+=1; continue
                frontier_support=min(unknown_count,known_free_count)
                d_robot=distance(self.est_xy,center)
                team_min=self._min_distance_to_points(center,team_points)
                team_pen=self._distance_band_penalty(team_min,hard_radius,soft_radius)
                intent_pen=self._known_teammate_region_penalty(center,radius)
                reserve_pen=self._reservation_overlap_penalty(center,reserved_targets,float(self.cfg.lidar.range)*float(self.cfg.planning.nbv_reservation_lidar_fraction))
                failed=self._failed_goal_penalty(center)
                known_support=min(1.0,known_free_count/max(1.0,0.45*block*block))
                clearance=float(clearance_map[cj,ci]) if 0<=cj<clearance_map.shape[0] and 0<=ci<clearance_map.shape[1] else 0.0
                clearance_score=min(1.5,clearance/max(1e-6,float(self.cfg.planning.desired_clearance_m)))
                score=(
                    2.3*math.log1p(float(unknown_count))+
                    0.85*math.sqrt(float(max(0,frontier_support)))+
                    0.65*known_support+
                    0.45*clearance_score-
                    0.18*d_robot-
                    5.5*team_pen-
                    6.5*intent_pen-
                    4.0*reserve_pen-
                    2.0*failed
                )
                regions.append(CoarseRegion((rid_i,rid_j),center,float(radius),unknown_count,int(known_free_count),int(frontier_support),float(score)))
                rid_i+=1
            rid_j+=1
        return regions

    def _known_teammate_region_penalty(self,center:Point,radius:float)->float:
        if not self.known_teammate_regions:
            return 0.0
        best=0.0
        for info in self.known_teammate_regions.values():
            task=str(info.get("task",""))
            if not (task.startswith("SEARCH") or task=="DEPLOY_FROM_HOME"):
                continue
            other_center=info.get("center")
            if other_center is None:
                continue
            other_radius=float(info.get("radius",0.0))
            overlap_radius=max(0.1,float(radius)+other_radius)
            d=distance(center,(float(other_center[0]),float(other_center[1])))
            if d<overlap_radius:
                best=max(best,1.0-d/overlap_radius)
        return float(best)

    def _region_unknown_mask(self,unknown_mask:np.ndarray,region:CoarseRegion)->np.ndarray:
        out=np.zeros_like(unknown_mask,dtype=bool)
        cell=self.map.world_to_cell(region.center)
        if cell is None:
            return unknown_mask
        ci,cj=cell
        radius=float(region.radius)+0.35*float(self.cfg.lidar.range)
        r_cells=max(1,int(math.ceil(radius/self.map.res)))
        y0,y1=max(0,cj-r_cells),min(self.map.ny,cj+r_cells+1)
        x0,x1=max(0,ci-r_cells),min(self.map.nx,ci+r_cells+1)
        for j in range(y0,y1):
            wy=(j+0.5)*self.map.res
            for i in range(x0,x1):
                wx=(i+0.5)*self.map.res
                if math.hypot(wx-region.center[0],wy-region.center[1])<=radius:
                    out[j,i]=True
        return unknown_mask & out

    def _select_next_best_view_goal(self,clearance_map:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]],dynamic_obstacles:list[tuple[Point,float]],reserved_targets:dict[int,Point]|None,relaxed:bool,focus_region:CoarseRegion|None=None)->tuple[float,Point,dict[str,float]]|None:
        team_points=self._team_history_points(team_paths,team_visits,team_trajectories)
        own_points=self._downsample_points(self.trajectory_from_home,max_points=90)
        hard_radius=float(self.cfg.planning.nbv_teammate_hard_avoid_lidar_fraction)*float(self.cfg.lidar.range)
        soft_radius=max(hard_radius,float(self.cfg.planning.nbv_teammate_soft_avoid_lidar_fraction)*float(self.cfg.lidar.range))
        own_radius=float(self.cfg.planning.nbv_own_path_avoid_lidar_fraction)*float(self.cfg.lidar.range)
        reservation_radius=float(self.cfg.planning.nbv_reservation_lidar_fraction)*float(self.cfg.lidar.range)
        active_unknown_mask=unknown_mask
        if focus_region is not None:
            active_unknown_mask=self._region_unknown_mask(unknown_mask,focus_region)
        candidates=self._build_nbv_scan_pose_candidates(clearance_map,known_mask,active_unknown_mask,occupied_mask,team_points,own_points,reserved_targets or {},hard_radius,soft_radius,own_radius,reservation_radius,relaxed,focus_region)
        return self._best_planned_nbv_candidate(candidates,dynamic_obstacles,team_points,own_points,hard_radius,soft_radius,own_radius,relaxed)

    def _build_nbv_scan_pose_candidates(self,clearance_map:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,team_points:list[Point],own_points:list[Point],reserved_targets:dict[int,Point],hard_radius:float,soft_radius:float,own_radius:float,reservation_radius:float,relaxed:bool,focus_region:CoarseRegion|None=None)->list[tuple[float,Point,dict[str,float]]]:
        stride=max(1,int(self.cfg.planning.nbv_sample_stride_cells))
        max_candidates=max(1,int(self.cfg.planning.nbv_max_candidates))
        desired=max(1e-6,float(self.cfg.planning.desired_clearance_m))
        safe_clearance=max(float(self.cfg.planning.safe_approach_min_clearance_m),float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m))
        local_radius=max(1,int(math.ceil(float(self.cfg.lidar.range)*float(self.cfg.planning.nbv_local_unknown_radius_lidar_fraction)/self.map.res)))
        min_goal_distance=max(0.75,float(self.cfg.robot.goal_tolerance)*1.5)
        focus_limit=math.inf
        if focus_region is not None:
            focus_limit=float(focus_region.radius)+0.95*float(self.cfg.lidar.range)
        pre:list[tuple[float,Point,dict[str,float]]]=[]
        blocked_by_teammate=0; blocked_by_reservation=0; skipped_low_gain=0; skipped_near=0
        # Robot-specific phase offset prevents every robot from testing exactly
        # the same lattice cells when they start with nearly identical maps.
        phase=(self.id*stride)//max(1,int(self.cfg.robot.count))
        for j in range(phase % stride,self.map.ny,stride):
            for i in range((phase+j) % stride,self.map.nx,stride):
                if occupied_mask[j,i]:
                    continue
                p=self.map.cell_to_world((i,j))
                d=distance(self.est_xy,p)
                region_focus=0.0
                if focus_region is not None:
                    d_region=distance(p,focus_region.center)
                    if d_region>focus_limit:
                        continue
                    region_focus=max(0.0,1.0-max(0.0,d_region-float(focus_region.radius))/max(1e-6,float(self.cfg.lidar.range)))
                if d<min_goal_distance:
                    skipped_near+=1
                    continue
                cl=float(clearance_map[j,i])
                if cl<safe_clearance:
                    continue
                local_unknown=self._local_cell_count(unknown_mask,i,j,local_radius)
                if local_unknown<4:
                    skipped_low_gain+=1
                    continue
                local_known=self._local_cell_count(known_mask & (~occupied_mask),i,j,local_radius)
                if local_known<2 and d>float(self.cfg.lidar.range)*0.85:
                    # This keeps the target inside/near the frontier of the
                    # unknown, not deep random unknown cells across the map.
                    continue
                team_min=self._min_distance_to_points(p,team_points)
                reserved_pen=self._reservation_overlap_penalty(p,reserved_targets,reservation_radius)
                if not relaxed and reserved_pen>0.0:
                    blocked_by_reservation+=1
                    continue
                if not relaxed and team_min<hard_radius:
                    blocked_by_teammate+=1
                    continue
                teammate_soft=self._distance_band_penalty(team_min,hard_radius,soft_radius)
                own_min=self._min_distance_to_points(p,own_points)
                own_soft=self._distance_band_penalty(own_min,0.0,max(0.1,own_radius))
                # Cheap pre-score before expensive LiDAR raycasting.
                unknown_density=min(2.0,local_unknown/45.0)
                known_support=min(1.0,local_known/20.0)
                target_inside_unknown=1.0 if bool(unknown_mask[j,i]) else 0.0
                clearance_score=min(1.5,cl/desired)
                failed=float(self._failed_goal_penalty(p))
                score=(
                    2.0*unknown_density+
                    0.75*target_inside_unknown+
                    0.65*known_support+
                    1.15*clearance_score+
                    0.75*region_focus-
                    0.28*d-
                    7.5*teammate_soft-
                    1.8*own_soft-
                    7.5*reserved_pen-
                    2.0*failed
                )
                bd={
                    "score":float(score),
                    "nbv_mode":float(2.0 if relaxed else 1.0),
                    "target_inside_unknown":float(target_inside_unknown),
                    "local_unknown_cells":float(local_unknown),
                    "local_known_free_cells":float(local_known),
                    "unknown_density_score":float(unknown_density),
                    "raw_clearance_m":float(cl),
                    "clearance_score":float(clearance_score),
                    "region_focus_score":float(region_focus),
                    "region_focus_reward":float(0.75*region_focus),
                    "distance_cost":float(0.28*d),
                    "teammate_min_distance_m":float(team_min if math.isfinite(team_min) else 9999.0),
                    "teammate_hard_avoid_radius_m":float(hard_radius),
                    "teammate_soft_avoid_radius_m":float(soft_radius),
                    "teammate_pose_penalty":float(7.5*teammate_soft),
                    "own_min_distance_m":float(own_min if math.isfinite(own_min) else 9999.0),
                    "own_path_penalty":float(1.8*own_soft),
                    "reservation_penalty":float(7.5*reserved_pen),
                    "failed_goal_penalty":float(2.0*failed),
                    "strict_candidates_blocked_by_teammate":float(blocked_by_teammate),
                    "strict_candidates_blocked_by_reservation":float(blocked_by_reservation),
                    "near_candidates_skipped":float(skipped_near),
                    "low_gain_candidates_skipped":float(skipped_low_gain),
                }
                pre.append((float(score),p,bd))
        if not pre:
            return []
        # Raycast only the best rough candidates; this keeps the GUI responsive.
        out=[]
        for prelim,p,bd in sorted(pre,key=lambda x:x[0],reverse=True)[:max_candidates]:
            expected_gain=self._expected_lidar_visibility_gain(p,unknown_mask,occupied_mask)
            if expected_gain<1.0:
                continue
            gain_score=math.log1p(expected_gain)
            final=prelim+2.6*gain_score
            info=dict(bd)
            info.update({
                "score":float(final),
                "pre_lidar_score":float(prelim),
                "expected_lidar_unknown_gain":float(expected_gain),
                "lidar_gain_score":float(gain_score),
                "lidar_gain_reward":float(2.6*gain_score),
            })
            out.append((float(final),p,info))
        return out

    def _best_planned_nbv_candidate(self,candidates:list[tuple[float,Point,dict[str,float]]],dynamic_obstacles:list[tuple[Point,float]],team_points:list[Point],own_points:list[Point],hard_radius:float,soft_radius:float,own_radius:float,relaxed:bool)->tuple[float,Point,dict[str,float]]|None:
        if not candidates:
            return None
        desired=max(1e-6,float(self.cfg.planning.desired_clearance_m))
        best=None
        eval_count=max(1,int(self.cfg.planning.nbv_plan_eval_count))
        for prelim,goal,bd in sorted(candidates,key=lambda x:x[0],reverse=True)[:eval_count]:
            result=self.planner.plan(self.map,self.est_xy,goal,dynamic_obstacles=dynamic_obstacles)
            if not result.success or len(result.path)<2:
                continue
            path_len=self._path_length(result.path)
            unknown_frac=self._path_unknown_fraction(result.path)
            path_team_overlap=self._path_history_overlap(result.path,team_points,hard_radius,soft_radius)
            path_own_overlap=self._path_history_overlap(result.path,own_points,0.0,max(0.1,own_radius))
            if (not relaxed) and path_team_overlap>0.70:
                continue
            clearance_bonus=1.0*min(1.5,float(result.min_clearance)/desired)
            unknown_path_bonus=0.55*unknown_frac
            detour_cost=0.20*path_len
            path_team_pen=7.0*path_team_overlap
            path_own_pen=1.2*path_own_overlap
            final=prelim+clearance_bonus+unknown_path_bonus-detour_cost-path_team_pen-path_own_pen
            out=dict(bd)
            out.update({
                "score":float(final),
                "pre_plan_score":float(prelim),
                "planned_path_length":float(path_len),
                "planned_path_clearance":float(result.min_clearance),
                "planned_path_unknown_fraction":float(unknown_frac),
                "path_clearance_bonus":float(clearance_bonus),
                "path_unknown_exploration_bonus":float(unknown_path_bonus),
                "path_detour_cost":float(detour_cost),
                "path_teammate_overlap":float(path_team_overlap),
                "path_own_overlap":float(path_own_overlap),
                "path_teammate_penalty":float(path_team_pen),
                "path_own_penalty":float(path_own_pen),
            })
            if best is None or final>best[0]:
                best=(float(final),goal,out)
        return best

    def _local_cell_count(self,mask:np.ndarray,i:int,j:int,radius:int)->int:
        y0=max(0,j-radius); y1=min(mask.shape[0],j+radius+1)
        x0=max(0,i-radius); x1=min(mask.shape[1],i+radius+1)
        return int(np.count_nonzero(mask[y0:y1,x0:x1]))

    def _exploration_safe_approach_point(self,frontier:FrontierCluster,clearance:np.ndarray,free:np.ndarray,known:np.ndarray)->Point:
        centroid=frontier.centroid_world
        ccell=self.map.world_to_cell(centroid)
        if ccell is None:
            return centroid
        radius=max(1,int(math.ceil(self.cfg.planning.safe_approach_search_radius_m/self.map.res)))
        ci,cj=ccell
        best_cell=None; best_score=-math.inf
        desired=max(1e-6,self.cfg.planning.desired_clearance_m)
        for j in range(max(0,cj-radius),min(self.map.ny,cj+radius+1)):
            for i in range(max(0,ci-radius),min(self.map.nx,ci+radius+1)):
                if not free[j,i] or not known[j,i]:
                    continue
                cl=float(clearance[j,i])
                if cl<self.cfg.planning.safe_approach_min_clearance_m:
                    continue
                p=self.map.cell_to_world((i,j))
                dc=distance(p,centroid)
                ds=distance(p,self.est_xy)
                # No sector ownership here.  Just choose a safe point near the
                # frontier boundary with good corridor center clearance.
                score=2.4*min(1.5,cl/desired)-0.38*dc-0.025*ds
                if score>best_score:
                    best_score=score; best_cell=(i,j)
        if best_cell is not None:
            return self.map.cell_to_world(best_cell)
        return self.map.safe_approach_point(frontier,self.est_xy,self.cfg.planning.safe_approach_search_radius_m,self.cfg.planning.safe_approach_min_clearance_m,self.cfg.planning.desired_clearance_m,clearance=clearance,free=free,known=known)

    def _team_history_points(self,team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]])->list[Point]:
        pts:list[Point]=[]
        for source in (team_paths,team_visits,team_trajectories):
            for rid,path in source.items():
                if rid==self.id or not path:
                    continue
                pts.extend(self._downsample_points(path,max_points=70))
        return self._downsample_points(pts,max_points=180)

    def _downsample_points(self,pts:list[Point],max_points:int)->list[Point]:
        if not pts:
            return []
        if len(pts)<=max_points:
            return [(float(x),float(y)) for x,y in pts]
        idx=np.linspace(0,len(pts)-1,max_points).astype(int)
        return [(float(pts[int(i)][0]),float(pts[int(i)][1])) for i in idx]

    def _min_distance_to_points(self,p:Point,pts:list[Point])->float:
        if not pts:
            return math.inf
        return float(min(distance(p,q) for q in pts))

    def _distance_band_penalty(self,d:float,hard_radius:float,soft_radius:float)->float:
        if not math.isfinite(d):
            return 0.0
        hard=max(0.0,float(hard_radius))
        soft=max(hard+1e-6,float(soft_radius))
        if d<=hard:
            return 1.0
        if d>=soft:
            return 0.0
        t=(soft-d)/(soft-hard)
        return float(max(0.0,min(1.0,t*t)))

    def _reservation_overlap_penalty(self,p:Point,reserved_frontiers:dict[int,Point],radius:float)->float:
        if not reserved_frontiers or radius<=1e-9:
            return 0.0
        best=0.0
        for rid,g in reserved_frontiers.items():
            if rid==self.id or g is None:
                continue
            d=distance(p,g)
            if d<radius:
                best=max(best,1.0-d/max(1e-6,radius))
        return float(best)

    def _path_history_overlap(self,path:list[Point],history:list[Point],hard_radius:float,soft_radius:float)->float:
        if not path or not history:
            return 0.0
        sampled_path=self._downsample_points(path,max_points=45)
        sampled_hist=self._downsample_points(history,max_points=140)
        vals=[]
        for p in sampled_path:
            d=self._min_distance_to_points(p,sampled_hist)
            vals.append(self._distance_band_penalty(d,hard_radius,soft_radius))
        return float(sum(vals)/max(1,len(vals)))

    def _path_unknown_fraction(self,path:list[Point])->float:
        if len(path)<2:
            return 1.0
        known=self.map.known_mask()
        cells=[]; seen=set()
        for a,b in zip(path[:-1],path[1:]):
            ca=self.map.world_to_cell(a); cb=self.map.world_to_cell(b)
            if ca is None or cb is None: continue
            for c in self.map._bresenham(ca,cb):
                if c not in seen:
                    seen.add(c); cells.append(c)
        if not cells:
            return 1.0
        unknown=sum(1 for i,j in cells if not known[j,i])
        return float(unknown)/float(len(cells))

    def _path_length(self,path:list[Point])->float:
        return float(sum(distance(a,b) for a,b in zip(path[:-1],path[1:]))) if len(path)>=2 else 0.0

    def _expected_lidar_visibility_gain(self,viewpoint:Point,unknown:np.ndarray,occupied:np.ndarray)->float:
        start=self.map.world_to_cell(viewpoint)
        if start is None:
            return 0.0
        rays=max(8,int(self.cfg.planning.frontier_visibility_rays))
        max_range=float(self.cfg.lidar.range)
        step_back=max(self.map.res,0.25)
        seen:set[tuple[int,int]]=set()
        gain=0.0
        for a in np.linspace(-math.pi,math.pi,rays,endpoint=False):
            rr=max_range
            end_cell=None
            while rr>self.map.res:
                end=(viewpoint[0]+math.cos(float(a))*rr,viewpoint[1]+math.sin(float(a))*rr)
                end_cell=self.map.world_to_cell(end)
                if end_cell is not None:
                    break
                rr-=step_back
            if end_cell is None:
                continue
            cells=self.map._bresenham(start,end_cell)
            for i,j in cells[1:]:
                if occupied[j,i]:
                    break
                if unknown[j,i] and (i,j) not in seen:
                    seen.add((i,j))
                    d=math.hypot((i-start[0])*self.map.res,(j-start[1])*self.map.res)
                    gain+=math.exp(-0.10*d)
        return float(gain)

    def _best_planned_frontier_candidate(self,candidates:list[tuple[float,Point,dict[str,float]]],dynamic_obstacles:list[tuple[Point,float]])->tuple[float,Point,dict[str,float]]|None:
        if not candidates:
            return None
        desired=max(1e-6,self.cfg.planning.desired_clearance_m)
        best=None
        eval_count=max(1,int(self.cfg.planning.frontier_plan_eval_count))
        for prelim,goal,bd in sorted(candidates,key=lambda x:x[0],reverse=True)[:eval_count]:
            result=self.planner.plan(self.map,self.est_xy,goal,dynamic_obstacles=dynamic_obstacles)
            if not result.success or len(result.path)<2:
                continue
            path_len=self._path_length(result.path)
            unknown_frac=self._path_unknown_fraction(result.path)
            clearance_bonus=self.cfg.planning.frontier_path_clearance_weight*min(1.5,result.min_clearance/desired)
            unknown_penalty=self.cfg.planning.frontier_path_unknown_penalty_weight*unknown_frac
            detour_cost=0.18*self.cfg.planning.distance_weight*path_len
            final=prelim+clearance_bonus-unknown_penalty-detour_cost
            out=dict(bd)
            out.update({
                "score":float(final),
                "pre_plan_score":float(prelim),
                "planned_path_length":float(path_len),
                "planned_path_clearance":float(result.min_clearance),
                "planned_path_unknown_fraction":float(unknown_frac),
                "path_clearance_bonus":float(clearance_bonus),
                "path_unknown_penalty":float(unknown_penalty),
                "path_detour_cost":float(detour_cost),
            })
            if best is None or final>best[0]:
                best=(float(final),goal,out)
        return best

    def _lidar_open_direction_reward(self,p:Point)->float:
        if self.assessment.open_sector_count<=0:
            return 0.0
        rel=wrap_angle(angle_to(self.est_xy,p)-self.est_pose[2])
        err=abs(wrap_angle(rel-self.assessment.best_open_angle))
        clearance_scale=float(np.clip(self.assessment.front_clearance/max(1e-6,self.cfg.lidar.range),0.15,1.0))
        return float(math.exp(-((err/0.75)**2))*clearance_scale)

    def _target_progress_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        cur=distance(self.est_xy,self.target.xy)
        cand=distance(p,self.target.xy)
        return float(np.clip((cur-cand)/max(1.0,self.cfg.lidar.range),-0.5,1.5))

    def _robot_target_corridor_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        sx,sy=self.est_xy; tx,ty=self.target.xy; px,py=p
        vx,vy=tx-sx,ty-sy; vv=vx*vx+vy*vy
        if vv<=1e-9:
            return 0.0
        t=max(0.0,min(1.0,((px-sx)*vx+(py-sy)*vy)/vv))
        cx,cy=sx+t*vx,sy+t*vy
        d=math.hypot(px-cx,py-cy)
        width=max(0.6,float(self.cfg.cage.target_corridor_width_m)*0.65)
        return float(math.exp(-((d/width)**2))*(0.35+0.65*t))

    def _target_probe_goal(self,target_goal:Point, allow_beyond_target:bool=False)->Point|None:
        base=angle_to(self.est_xy,target_goal)
        offsets=[0.0,math.radians(14),-math.radians(14),math.radians(28),-math.radians(28),math.radians(45),-math.radians(45)]
        if self.assessment.blocked_forward or self.assessment.front_clearance<self.cfg.planning.critical_clearance_m:
            offsets=[self.assessment.best_open_angle]+offsets
        best=None; best_score=-math.inf
        max_step=min(self.cfg.lidar.range*0.70,3.4,distance(self.est_xy,target_goal))
        if max_step<=self.cfg.robot.goal_tolerance:
            if not allow_beyond_target:
                return target_goal
            base=self.est_pose[2]+self.assessment.best_open_angle
            max_step=min(self.cfg.lidar.range*0.55,2.4)
        for dist_m in np.linspace(max(0.9,self.cfg.robot.goal_tolerance*2.0),max_step,5):
            for off in offsets:
                a=base+off
                p=(self.est_xy[0]+math.cos(a)*float(dist_m),self.est_xy[1]+math.sin(a)*float(dist_m))
                cell=self.map.world_to_cell(p)
                if cell is None:
                    continue
                cl=self.map.clearance_at(p,max_radius_m=2.5)
                if cl<self.cfg.planning.critical_clearance_m:
                    continue
                progress=self._target_progress_reward(p)
                corridor=self._robot_target_corridor_reward(p)
                score=6.0*progress+2.2*corridor+min(1.5,cl/max(1e-6,self.cfg.planning.desired_clearance_m))-0.04*distance(self.est_xy,p)
                if score>best_score:
                    best_score=score; best=(float(p[0]),float(p[1]))
        return best

    def _target_corridor_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        hx,hy=self.home_xy; tx,ty=self.target.xy; px,py=p
        vx,vy=tx-hx,ty-hy; vv=vx*vx+vy*vy
        if vv<=1e-9:
            return 0.0
        t=max(0.0,min(1.0,((px-hx)*vx+(py-hy)*vy)/vv))
        cx,cy=hx+t*vx,hy+t*vy
        d=math.hypot(px-cx,py-cy)
        width=max(0.5,float(self.cfg.cage.target_corridor_width_m))
        tube=math.exp(-((d/width)**2))
        along=0.45+0.55*t
        return float(tube*along)

    def _target_corridor_low_quality_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        base=self._target_corridor_reward(p)
        if base<=1e-6:
            return 0.0
        cell=self.map.world_to_cell(p)
        if cell is None:
            return 0.0
        i,j=cell
        radius=max(1,int(round(1.2/self.map.res)))
        y0,y1=max(0,j-radius),min(self.map.ny,j+radius+1)
        x0,x1=max(0,i-radius),min(self.map.nx,i+radius+1)
        known=self.map.known_mask()[y0:y1,x0:x1]
        quality=np.clip(self.map.quality[y0:y1,x0:x1],0.0,1.0)
        if known.size==0:
            return 0.0
        unknown_frac=1.0-float(np.count_nonzero(known))/float(known.size)
        mean_q=float(np.mean(quality[known])) if np.any(known) else 0.0
        lowq=max(0.0,1.0-mean_q)
        return float(base*(0.65*unknown_frac+0.35*lowq))

    def _should_report_target_to_home(self)->bool:
        if self.target.source_robot==self.id: return True
        if any(task=="REPORT_TARGET_HOME" for task in self.known_teammate_tasks.values()): return False
        my_home_d=distance(self.est_xy,self.home_xy)
        teammate_home_distances=[
            distance((float(pose[0]),float(pose[1])),self.home_xy)
            for pose in self.known_teammate_pose.values()
        ]
        return not teammate_home_distances or my_home_d<=min(teammate_home_distances)+1.5
    def _best_local_target_route(self):
        routes=self.graph.top_routes(k=1)
        return routes[0] if routes else None
    def _failed_goal_penalty(self,p:Point)->float:
        return math.exp(-min(distance(p,q) for q in self.failed_goal_memory)/2.2) if self.failed_goal_memory else 0.0
    def _sector_probe_goal(self)->Point|None:
        mission_angle=angle_to(self.home_xy,self.search_prior_xy)
        max_d=min(self.cfg.lidar.range*0.75,3.8)
        for dist_m in np.linspace(max(1.4,self.cfg.robot.goal_tolerance*2.5),max_d,6):
            for off in (0.0,math.radians(18),-math.radians(18),math.radians(36),-math.radians(36)):
                a=mission_angle+off
                p=(self.est_xy[0]+math.cos(a)*float(dist_m),self.est_xy[1]+math.sin(a)*float(dist_m))
                cell=self.map.world_to_cell(p)
                if cell is None:
                    continue
                cl=self.map.clearance_at(p,max_radius_m=2.5)
                if cl>=self.cfg.planning.critical_clearance_m:
                    return (float(p[0]),float(p[1]))
        return None

    def _nearest_anchor(self)->Point|None:
        anchors=[n.xy for n in self.graph.nodes.values() if n.kind in {"home","anchor"}]
        return min(anchors,key=lambda p:distance(self.est_xy,p)) if anchors else None
    def _downsample_path(self,path:list[Point],spacing:float)->list[Point]:
        if len(path)<=2: return path
        out=[path[0]]; last=path[0]
        for p in path[1:-1]:
            if distance(last,p)>=spacing: out.append(p); last=p
        out.append(path[-1]); return out
    def _target_fallback_path(self,goal:Point)->list[Point]:
        if goal is None:
            return []
        d=distance(self.est_xy,goal)
        if d<=self.cfg.robot.goal_tolerance:
            return []
        probe=self._target_probe_goal(goal)
        if probe is None:
            return []
        return [self.est_xy,probe]

    def _homing_fallback_path(self,goal:Point)->list[Point]:
        d=distance(self.est_xy,goal)
        if d<=self.cfg.robot.goal_tolerance: return []
        desired=angle_to(self.est_xy,goal)
        if self.assessment.blocked_forward or self.assessment.front_clearance<self.cfg.planning.critical_clearance_m:
            desired=self.est_pose[2]+self.assessment.best_open_angle
        step=min(1.6,max(0.6,d))
        p=(self.est_xy[0]+math.cos(desired)*step,self.est_xy[1]+math.sin(desired)*step)
        cell=self.map.world_to_cell(p)
        if cell is None:
            return []
        return [self.est_xy,p]
    def _teammate_avoidance_control(self)->tuple[float,float]:
        turn=0.0
        speed_scale=1.0
        base=2.0*float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m)
        horizon=max(0.1,float(self.cfg.robot.collision_avoidance_horizon_m))
        for rid,pose in self.known_teammate_pose.items():
            if rid==self.id:
                continue
            mate=(float(pose[0]),float(pose[1]))
            d=distance(self.est_xy,mate)
            cov=float(self.known_teammate_cov.get(rid,0.0))
            cov_extra=min(float(self.cfg.planning.dynamic_obstacle_max_cov_extra_m),0.55*math.sqrt(max(0.0,cov)*0.5))
            hard=base+cov_extra
            slow_radius=hard+horizon
            if d>=slow_radius:
                continue
            rel=wrap_angle(angle_to(self.est_xy,mate)-self.est_pose[2])
            strength=float(np.clip((slow_radius-d)/horizon,0.0,1.0))
            if abs(rel)<math.radians(115.0):
                side=-1.0 if rel>=0.0 else 1.0
                if abs(rel)<0.08:
                    side=-1.0 if rid<self.id else 1.0
                turn+=side*float(self.cfg.robot.teammate_avoidance_turn_gain)*strength
            if abs(rel)<math.radians(70.0):
                clear=max(0.0,d-hard)
                speed_scale=min(speed_scale,float(np.clip(clear/max(0.1,horizon),0.0,1.0)))
        return float(np.clip(speed_scale,0.0,1.0)),float(turn)
    def compute_control(self)->tuple[float,float]:
        if not self.path or self.path_index>=len(self.path): self.last_command=(0.0,0.0); return self.last_command
        pos=self.est_xy; target=self.path[self.path_index]
        if distance(pos,target)<self.cfg.robot.waypoint_tolerance:
            self.path_index+=1
            if self.path_index>=len(self.path): self.last_command=(0.0,0.0); return self.last_command
            target=self.path[self.path_index]
        desired=angle_to(pos,target); err=wrap_angle(desired-self.est_pose[2])
        side_bias=0.0; side_diff=self.assessment.right_clearance-self.assessment.left_clearance
        if min(self.assessment.left_clearance,self.assessment.right_clearance)<self.cfg.planning.desired_clearance_m:
            side_bias=0.35*np.clip(side_diff/max(1e-6,self.cfg.planning.desired_clearance_m),-1.0,1.0)
        avoid_speed_scale,avoid_turn=self._teammate_avoidance_control()
        omega=float(np.clip(self.cfg.robot.turn_gain*err+side_bias+avoid_turn,-1.4,1.4))
        consistency_scale=0.45 if self.assessment.consistency<self.cfg.assessment.caution_consistency else 1.0
        front_scale=np.clip((self.assessment.front_clearance-self.cfg.lidar.blocked_forward_distance)/1.25,0.0,1.0)
        side_scale=np.clip(min(self.assessment.left_clearance,self.assessment.right_clearance)/max(1e-6,self.cfg.planning.desired_clearance_m),0.25,1.0)
        v=self.cfg.robot.max_speed*consistency_scale*max(0.12,float(front_scale))*side_scale*max(0.20,math.cos(err))*avoid_speed_scale
        if self.assessment.blocked_forward: v=0.0
        self.last_command=(float(v),omega); return self.last_command
