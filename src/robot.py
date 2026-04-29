"""Robot agent for the clean Search-CAGE baseline.

Every robot plans from its own local LiDAR-built map, its EKF pose estimate,
and packet-received teammate intent.  Ground truth is used only by the
simulator to produce sensing/collision/evaluation.
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
    estimated_pose: tuple[float, float, float]; pose_cov_trace: float

@dataclass
class RobotStatus:
    task: str = "INIT"; planning_source: str = "local LiDAR map + EKF pose estimate"
    note: str = ""; goal: Point | None = None; last_plan_success: bool = False; last_plan_reason: str = ""
    last_path_min_clearance: float = 0.0; reward_breakdown: dict[str, float] = field(default_factory=dict)

class RobotAgent:
    def __init__(self, robot_id: int, initial_pose: Pose, cfg: AppConfig, world: World, rng: np.random.Generator):
        self.id=robot_id; self.cfg=cfg; self.rng=rng
        self.true_pose=np.array(initial_pose,dtype=float); self.true_path=[(float(initial_pose[0]),float(initial_pose[1]))]
        self.estimator=PoseEstimator(initial_pose,cfg.motion,rng); self.lidar=LidarSensor(cfg.lidar,rng)
        self.map=OccupancyGrid(world.width,world.height,cfg.mapping); self.graph=RouteGraph(cfg.cage.edge_merge_distance)
        self.home_node=self.graph.add_node(world.home,kind="home",confidence=1.0,allow_merge=False)
        self.home_xy=world.home
        self.search_prior_xy=self._sector_prior_point(world)
        self.last_graph_node=self.home_node; self.last_keypoint_xy=world.home
        self.scan: LidarScan | None=None; self.assessment=LidarAssessment(); self.planner=GridPlanner(cfg.planning)
        self.path=[]; self.path_index=0; self.last_replan_time=-999.0; self.current_goal=None; self.current_task="SEARCH"
        self.goal_commit_start=-999.0; self.goal_commit_score=-math.inf; self.best_goal_distance=math.inf; self.last_goal_progress_time=0.0
        self.status=RobotStatus(task="SEARCH"); self.target=TargetReport()
        self.known_teammate_goals={}; self.known_teammate_paths={}; self.known_teammate_visits={}; self.known_teammate_tasks={}
        self.known_teammate_pose={}; self.known_teammate_cov={}; self.known_teammate_last_seen={}
        self.visit_history=[self.est_xy]; self.failed_goal_memory=[]
        self.last_command=(0.0,0.0); self.last_pose_quality=1.0; self.best_routes=[]; self.received_packets=0; self.blocked_events=0

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

    def step_predict_and_move(self, world: World, dt: float) -> None:
        v,omega=self.last_command; x,y,th=self.true_pose
        new_th=wrap_angle(th+omega*dt); cand=(float(x+math.cos(new_th)*v*dt), float(y+math.sin(new_th)*v*dt))
        executed_v, executed_omega = v, omega
        if world.is_free(cand, margin=self.cfg.robot.radius): self.true_pose[:]=[cand[0],cand[1],new_th]
        else:
            self.blocked_events+=1; self.path=[]; self.status.note="true_collision_prevented_by_sim"; self.last_command=(0.0,0.0)
            executed_v, executed_omega = 0.0, 0.0
        self._append_true_path_sample()
        self.estimator.predict_from_command(executed_v,executed_omega,dt)

    def _append_true_path_sample(self)->None:
        xy=(float(self.true_pose[0]),float(self.true_pose[1]))
        if self.true_path and distance(xy,self.true_path[-1])<self.cfg.robot.true_path_spacing_m:
            return
        self.true_path.append(xy)
        if len(self.true_path)>self.cfg.robot.max_true_path_points:
            del self.true_path[:-self.cfg.robot.max_true_path_points]

    def sense_update_map_and_belief(self, world: World, time_s: float) -> None:
        visible_landmarks=world.visible_landmarks(tuple(self.true_pose), self.cfg.world.landmark_detection_range)
        self.estimator.update_with_landmarks(visible_landmarks, self.cfg.world.landmark_detection_range)
        self.scan=self.lidar.sense(world, tuple(self.true_pose)); self.last_pose_quality=self.estimator.quality()
        prev=None if self.assessment.decision_note=="init" else self.assessment.consistency
        self.map.update_from_lidar(self.est_pose,self.scan,self.last_pose_quality,self.id,time_s)
        self.assessment=assess_lidar(self.map,self.est_pose,self.scan,self.cfg.lidar,self.cfg.assessment,prev)
        self._update_visit_history(); self._detect_target(world,time_s); self._update_route_graph(time_s)

    def _update_visit_history(self)->None:
        xy=self.est_xy
        if not self.visit_history or distance(xy,self.visit_history[-1])>=self.cfg.robot.visit_history_spacing_m:
            self.visit_history.append(xy); self.visit_history=self.visit_history[-self.cfg.robot.max_visit_history:]

    def _detect_target(self, world: World, time_s: float)->None:
        if self.target.detected or not world.target_visible(tuple(self.true_pose), self.cfg.lidar.range): return
        true_xy=(float(self.true_pose[0]),float(self.true_pose[1])); brg=angle_to(true_xy,world.target)-float(self.true_pose[2]); rr=distance(true_xy,world.target)
        r=max(0.05,rr+self.rng.normal(0.0,0.06)); b=brg+self.rng.normal(0.0,math.radians(2.0))
        ex,ey,eth=self.est_pose; est_target=(float(ex+math.cos(eth+b)*r), float(ey+math.sin(eth+b)*r))
        conf=float(np.clip(self.assessment.consistency*self.last_pose_quality,0.1,1.0))
        self.target=TargetReport(True,est_target,conf,self.id,time_s,False)
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
        self.received_packets+=1; self.map.merge_from_digest(packet.map_digest); self.graph.merge_from_digest(packet.graph_digest)
        if packet.sender_id>=0:
            self.known_teammate_pose[packet.sender_id]=packet.estimated_pose; self.known_teammate_cov[packet.sender_id]=float(packet.pose_cov_trace)
            self.known_teammate_last_seen[packet.sender_id]=float(packet.time_s); self.known_teammate_tasks[packet.sender_id]=packet.task
            if packet.current_goal is not None: self.known_teammate_goals[packet.sender_id]=(float(packet.current_goal[0]),float(packet.current_goal[1]))
            else: self.known_teammate_goals.pop(packet.sender_id,None)
            self.known_teammate_paths[packet.sender_id]=[(float(x),float(y)) for x,y in packet.current_path_digest]
            self.known_teammate_visits[packet.sender_id]=[(float(x),float(y)) for x,y in packet.visited_digest]
        if packet.target_report and packet.target_report.get("detected"):
            tr=packet.target_report; conf=float(tr.get("confidence",0.0))
            if not self.target.detected or conf>self.target.confidence:
                xy=tuple(tr["xy"]); self.target=TargetReport(True,(float(xy[0]),float(xy[1])),conf,int(tr.get("source_robot",packet.sender_id)),float(tr.get("time_s",packet.time_s)),bool(tr.get("reported_home",False)))
                tid=self.graph.add_node(self.target.xy,kind="target",confidence=conf,allow_merge=True); self.graph.target_id=tid
            elif bool(tr.get("reported_home",False)):
                self.target.reported_home=True
        self._expire_stale_teammate_intent(packet.time_s)

    def make_packet(self,time_s:float)->RobotPacket:
        target_dict=None
        if self.target.detected and self.target.xy is not None:
            target_dict={"detected":True,"xy":[float(self.target.xy[0]),float(self.target.xy[1])],"confidence":float(self.target.confidence),"source_robot":int(self.target.source_robot),"time_s":float(self.target.time_s),"reported_home":bool(self.target.reported_home)}
        return RobotPacket(self.id,float(time_s),self.map.make_digest(self.id,time_s),self.graph.make_digest(self.id,time_s),target_dict,self.current_task,self.current_goal,self._path_digest(),self._visited_digest(),self.est_pose,self.cov_trace)

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

    def choose_task_and_plan(self,time_s:float)->None:
        self._expire_stale_teammate_intent(time_s); team_goals=self.fresh_teammate_goals(time_s); team_paths=self.fresh_teammate_paths(time_s); team_visits=self.fresh_teammate_visits(time_s)
        event_replan=False
        if self.assessment.blocked_forward and self.path:
            self._remember_failed_goal(); self.path=[]; self.path_index=0; self.status.note="path_invalidated_by_lidar_block"; event_replan=True
        if self._goal_progress_stalled(time_s):
            self._remember_failed_goal(); self.path=[]; self.path_index=0; self.status.note="goal_progress_stalled"; event_replan=True
        if self._should_keep_committed_goal(time_s,event_replan): return
        if not event_replan and time_s-self.last_replan_time<self.cfg.robot.path_replan_period_s:
            goal_reached=self.current_goal is not None and distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_tolerance
            if self.path_index<len(self.path) or not goal_reached:
                return
        goal,task,reason,breakdown=self._select_goal_from_lidar_map(team_goals,team_paths,team_visits)
        if self._should_reject_goal_switch(goal,breakdown,time_s,event_replan):
            self.status.note="committed_current_goal"
            self.last_replan_time=time_s
            return
        self.current_goal=goal; self.current_task=task; self.status.task=task; self.status.goal=goal
        self.status.planning_source="LOCAL LiDAR map + EKF pose estimate + packet intent only"; self.status.note=reason; self.status.reward_breakdown=breakdown
        if goal is None:
            self.path=[]; self.path_index=0; self.status.last_plan_success=False; self.status.last_plan_reason="no_goal_available"; self.status.last_path_min_clearance=0.0; self.last_replan_time=time_s; return
        result=self.planner.plan(self.map,self.est_xy,goal)
        if result.success and len(result.path)>=2:
            simplified=self._downsample_path(result.path,spacing=0.45); simp_clear=self.map.path_min_clearance(simplified)
            self.path=simplified if simp_clear>=self.cfg.planning.critical_clearance_m else result.path; self.path_index=0
            self.status.last_path_min_clearance=max(0.0,min(result.min_clearance,simp_clear if simplified else result.min_clearance))
        elif task in {"REPORT_TARGET_HOME","RETURN_HOME_CERT_ROUTE"}:
            self.path=self._homing_fallback_path(goal); self.path_index=0
            result.success=bool(self.path); result.reason="homing_fallback" if self.path else result.reason
            self.status.last_path_min_clearance=max(0.0,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        else:
            self._remember_failed_goal(goal); self.path=[]; self.path_index=0; self.status.last_path_min_clearance=0.0
        self.status.last_plan_success=result.success; self.status.last_plan_reason=result.reason; self.last_replan_time=time_s; self.best_routes=self.graph.top_routes(k=4)
        if result.success:
            self.goal_commit_start=time_s; self.goal_commit_score=float(breakdown.get("score",0.0)); self.best_goal_distance=distance(self.est_xy,goal); self.last_goal_progress_time=time_s

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
        if distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_tolerance: return False
        if time_s-self.goal_commit_start<self.cfg.robot.goal_commit_time_s:
            self.status.note="commit_hold"
            return True
        return False

    def _should_reject_goal_switch(self,goal:Point|None,breakdown:dict[str,float],time_s:float,event_replan:bool)->bool:
        if event_replan or goal is None or self.current_goal is None or not self.path or self.path_index>=len(self.path): return False
        if self.target.detected or distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_tolerance: return False
        if distance(goal,self.current_goal)<0.85: return False
        new_score=float(breakdown.get("score",0.0))
        return new_score<self.goal_commit_score+self.cfg.robot.goal_switch_score_margin

    def fresh_teammate_goals(self,time_s:float)->dict[int,Point]: self._expire_stale_teammate_intent(time_s); return dict(self.known_teammate_goals)
    def fresh_teammate_paths(self,time_s:float)->dict[int,list[Point]]: self._expire_stale_teammate_intent(time_s); return {rid:list(path) for rid,path in self.known_teammate_paths.items() if path}
    def fresh_teammate_visits(self,time_s:float)->dict[int,list[Point]]: self._expire_stale_teammate_intent(time_s); return {rid:list(path) for rid,path in self.known_teammate_visits.items() if path}
    def _expire_stale_teammate_intent(self,time_s:float)->None:
        stale=[rid for rid,stamp in self.known_teammate_last_seen.items() if time_s-stamp>self.cfg.communication.teammate_intent_timeout_s]
        for rid in stale:
            for d in (self.known_teammate_last_seen,self.known_teammate_goals,self.known_teammate_paths,self.known_teammate_visits,self.known_teammate_tasks,self.known_teammate_pose,self.known_teammate_cov): d.pop(rid,None)

    def _select_goal_from_lidar_map(self,team_goals:dict[int,Point],team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]])->tuple[Point|None,str,str,dict[str,float]]:
        if self.target.detected and self.target.xy is not None:
            if not self.target.reported_home and self._should_report_target_to_home():
                return self.home_xy,"REPORT_TARGET_HOME","target_known_return_to_home_report",{"report_target":1.0,"distance":distance(self.est_xy,self.home_xy)}
            if self.target.reported_home:
                home_dist=distance(self.est_xy,self.home_xy)
                route = self._best_local_target_route()
                if home_dist>self.cfg.robot.goal_tolerance:
                    cert = route.certificate if route is not None else 0.0
                    return self.home_xy,"RETURN_HOME_CERT_ROUTE","home_has_target_report_return_to_base",{"route_cert":cert,"distance":home_dist}
                return None,"WAIT_AT_HOME","home_has_target_report_wait_at_base",{"distance_home":home_dist}
            dtarget=distance(self.est_xy,self.target.xy); bd={"target_progress":max(0.0,8.0-dtarget),"distance":dtarget}
            return (self.target.xy,"ADVANCE_TO_TARGET","target_known_plan_to_detected_target",bd) if dtarget>self.cfg.robot.goal_tolerance else (self.target.xy,"CERTIFY_TARGET_EDGE","near_detected_target_certifying_edge",bd)
        if self.assessment.consistency<self.cfg.cage.reanchor_consistency_threshold:
            anchor=self._nearest_anchor()
            if anchor is not None and distance(anchor,self.est_xy)>0.35: return anchor,"REANCHOR","low_scan_map_consistency_reanchor",{"consistency":self.assessment.consistency}
        frontiers=self.map.find_frontiers(self.cfg.planning.frontier_min_cluster_size,self.cfg.planning.frontier_info_radius_m)
        if not frontiers:
            gx=self.est_xy[0]+math.cos(self.est_pose[2]+self.assessment.best_open_angle)*min(1.8,self.cfg.lidar.range*0.38); gy=self.est_xy[1]+math.sin(self.est_pose[2]+self.assessment.best_open_angle)*min(1.8,self.cfg.lidar.range*0.38)
            return (gx,gy),"SEARCH_OPEN_SECTOR","no_frontier_use_best_lidar_open_sector",{"open_sector":1.0}
        clearance_map=self.map.clearance_map(max_radius_m=max(3.0,self.cfg.planning.desired_clearance_m*2.5))
        free_mask=self.map.free_mask(); known_mask=self.map.known_mask()
        best=None; skipped_near=[]
        for fr in frontiers[:max(1,self.cfg.planning.frontier_sample_count)]:
            approach=self.map.safe_approach_point(fr,self.est_xy,self.cfg.planning.safe_approach_search_radius_m,self.cfg.planning.safe_approach_min_clearance_m,self.cfg.planning.desired_clearance_m,clearance=clearance_map,free=free_mask,known=known_mask)
            d=max(0.1,distance(self.est_xy,approach)); cell=self.map.world_to_cell(approach)
            clearance=float(clearance_map[cell[1],cell[0]]) if cell is not None else 0.0
            if d < max(0.75,self.cfg.robot.goal_tolerance*1.5):
                skipped_near.append((fr,approach,d,clearance))
                continue
            info=self.cfg.planning.information_weight*math.log1p(fr.information_gain); ratio=min(1.0,clearance/max(1e-6,self.cfg.planning.desired_clearance_m))
            clear_r=self.cfg.planning.clearance_weight*min(2.5,clearance); center=self.cfg.planning.centerline_weight*(ratio**2)
            recent=self._recent_visit_penalty(approach); teammate_recent=self._teammate_visit_penalty(approach,team_visits); failed=self._failed_goal_penalty(approach); tg=self._teammate_goal_penalty(approach,team_goals); tp=self._teammate_path_penalty(approach,team_paths); re=self._route_extension_reward(approach); prior=self._search_prior_reward(approach)
            score=info+clear_r+center+self.cfg.planning.route_alternate_weight*re+self.cfg.planning.goal_progress_weight*prior-self.cfg.planning.distance_weight*d-self.cfg.planning.recent_visit_penalty_weight*(recent+0.55*teammate_recent+1.4*failed)-self.cfg.planning.duplicate_penalty_weight*tg-self.cfg.planning.teammate_path_penalty_weight*tp
            bd={"score":float(score),"info":float(info),"center":float(center),"raw_clearance_m":float(clearance),"distance_cost":float(self.cfg.planning.distance_weight*d),"recent_penalty":float(self.cfg.planning.recent_visit_penalty_weight*recent),"teammate_recent_penalty":float(self.cfg.planning.recent_visit_penalty_weight*0.55*teammate_recent),"failed_goal_penalty":float(self.cfg.planning.recent_visit_penalty_weight*1.4*failed),"teammate_goal_penalty":float(self.cfg.planning.duplicate_penalty_weight*tg),"teammate_path_penalty":float(self.cfg.planning.teammate_path_penalty_weight*tp),"route_ext":float(re),"search_prior":float(self.cfg.planning.goal_progress_weight*prior)}
            if best is None or score>best[0]: best=(score,approach,bd)
        if best is None and skipped_near:
            fr,approach,d,clearance=max(skipped_near,key=lambda x:x[0].information_gain)
            best=(0.0,approach,{"score":0.0,"info":float(fr.information_gain),"raw_clearance_m":float(clearance),"distance_cost":float(self.cfg.planning.distance_weight*d),"near_fallback":1.0})
        if best is None: return None,"WAIT","no_good_frontier",{}
        return best[1],"SEARCH_FRONTIER","depth_frontier_selected_local_map_only",best[2]

    def _teammate_goal_penalty(self,p:Point,team_goals:dict[int,Point])->float:
        pen=0.0
        my_d=distance(self.est_xy,p)
        for rid,g in team_goals.items():
            if rid==self.id or g is None: continue
            dg=distance(g,p)
            if dg>6.0: continue
            base=math.exp(-dg/2.2)
            mate_pose=self.known_teammate_pose.get(rid)
            if mate_pose is not None:
                mate_xy=(float(mate_pose[0]),float(mate_pose[1]))
                mate_d=distance(mate_xy,g)
                owner_bonus=1.65 if mate_d<=my_d+1.5 else 0.65
            else:
                owner_bonus=1.0
            pen+=owner_bonus*base
        return min(2.5,pen)
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
    def _teammate_path_penalty(self,p:Point,team_paths:dict[int,list[Point]])->float:
        pen=0.0
        for rid,pts in team_paths.items():
            if rid==self.id: continue
            path_pen=0.0
            for q in pts:
                d=distance(p,q)
                if d<5.0: path_pen=max(path_pen,math.exp(-d/1.35))
            pen+=path_pen
        return min(2.0,pen)
    def _recent_visit_penalty(self,p:Point)->float:
        return math.exp(-min(distance(p,q) for q in self.visit_history[-100:])/2.0) if self.visit_history else 0.0
    def _teammate_visit_penalty(self,p:Point,team_visits:dict[int,list[Point]])->float:
        best=0.0
        for rid,pts in team_visits.items():
            if rid==self.id or not pts: continue
            best=max(best,math.exp(-min(distance(p,q) for q in pts)/2.4))
        return best
    def _failed_goal_penalty(self,p:Point)->float:
        return math.exp(-min(distance(p,q) for q in self.failed_goal_memory)/2.2) if self.failed_goal_memory else 0.0
    def _route_extension_reward(self,p:Point)->float:
        pts=[n.xy for n in self.graph.nodes.values() if n.kind in {"home","anchor","keypoint"}]
        return min(3.0,min(distance(p,q) for q in pts)/3.0) if pts else 0.0
    def _search_prior_reward(self,p:Point)->float:
        home_depth=distance(self.home_xy,self.est_xy); cand_depth=distance(self.home_xy,p)
        depth_gain=max(-1.0,min(4.0,cand_depth-home_depth))
        prior_progress=max(-2.0,min(5.0,distance(self.est_xy,self.search_prior_xy)-distance(p,self.search_prior_xy)))
        mission_angle=angle_to(self.home_xy,self.search_prior_xy); cand_angle=angle_to(self.home_xy,p)
        bearing=math.exp(-abs(wrap_angle(cand_angle-mission_angle))/0.62)
        normalized_depth=min(1.0,cand_depth/max(1e-6,distance(self.home_xy,self.search_prior_xy)))
        return 0.65*depth_gain+1.05*prior_progress+1.2*bearing+self.cfg.cage.unknown_target_search_bias*normalized_depth
    def _nearest_anchor(self)->Point|None:
        anchors=[n.xy for n in self.graph.nodes.values() if n.kind in {"home","anchor"}]
        return min(anchors,key=lambda p:distance(self.est_xy,p)) if anchors else None
    def _downsample_path(self,path:list[Point],spacing:float)->list[Point]:
        if len(path)<=2: return path
        out=[path[0]]; last=path[0]
        for p in path[1:-1]:
            if distance(last,p)>=spacing: out.append(p); last=p
        out.append(path[-1]); return out
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
        omega=float(np.clip(self.cfg.robot.turn_gain*err+side_bias,-1.4,1.4))
        consistency_scale=0.45 if self.assessment.consistency<self.cfg.assessment.caution_consistency else 1.0
        front_scale=np.clip((self.assessment.front_clearance-self.cfg.lidar.blocked_forward_distance)/1.25,0.0,1.0)
        side_scale=np.clip(min(self.assessment.left_clearance,self.assessment.right_clearance)/max(1e-6,self.cfg.planning.desired_clearance_m),0.25,1.0)
        v=self.cfg.robot.max_speed*consistency_scale*max(0.12,float(front_scale))*side_scale*max(0.20,math.cos(err))
        if self.assessment.blocked_forward: v=0.0
        self.last_command=(float(v),omega); return self.last_command
