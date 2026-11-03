#!/usr/bin/env python3
import heapq
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Set, List, Iterable, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
import sys

Coord = Tuple[float, float]
Index = Tuple[int, int]
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# ---------------- Grid graph ----------------
@dataclass
class GridGraph:
    """
    Lightweight grid graph whose nodes are (row, col) indices mapped
    to metric (x, y) world coordinates. Edges connect 4-neighborhood cells.
    """
    H: int
    W: int
    positions: Dict[Index, Coord]   # (r,c) -> (x,y)
    coord2idx: Dict[Coord, Index]   # (x,y) -> (r,c)
    adj: Dict[Index, Set[Index]]    # adjacency

    @classmethod
    def from_node_graph(cls, node_graph: List[List[List[float]]]):
        """
        Build a GridGraph from a 2D list of XY coordinates.

        Args:
            node_graph: HxW list; node_graph[r][c] is [x, y] for grid cell (r, c).

        Returns:
            GridGraph: A fully connected 4-neighborhood graph with
            (r,c)<->(x,y) mappings initialized.
        """
        H, W = len(node_graph), len(node_graph[0])
        positions: Dict[Index, Coord] = {}
        coord2idx: Dict[Coord, Index] = {}
        for r in range(H):
            for c in range(W):
                xy = tuple(float(v) for v in node_graph[r][c])
                positions[(r, c)] = xy
                coord2idx[xy] = (r, c)
        adj: Dict[Index, Set[Index]] = {(r, c): set() for r in range(H) for c in range(W)}
        for r in range(H):
            for c in range(W):
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        adj[(r, c)].add((nr, nc))
                        adj[(nr, nc)].add((r, c))
        return cls(H, W, positions, coord2idx, adj)

    def index_of(self, coord: Coord, eps: float = 1e-9) -> Index:
        """
        Look up the grid index for a given (x, y) coordinate.

        Args:
            coord: (x, y) world coordinate.
            eps: absolute tolerance for fallback floating-point comparison.

        Returns:
            (r, c) integer grid index.

        Raises:
            KeyError: if the coordinate is not found (within tolerance).
        """
        x, y = float(coord[0]), float(coord[1])
        key = (x, y)
        if key in self.coord2idx:
            return self.coord2idx[key]
        for idx, (px, py) in self.positions.items():
            if math.isclose(px, x, abs_tol=eps) and math.isclose(py, y, abs_tol=eps):
                return idx
        raise KeyError(f"Coordinate {coord} not found.")

    def coord_of(self, idx: Index) -> Coord:
        """
        Get the (x, y) coordinate for a given grid index.

        Args:
            idx: (r, c) integer grid index.

        Returns:
            (x, y) world coordinate.
        """
        return self.positions[idx]

    def has_edge(self, a: Coord, b: Coord) -> bool:
        """
        Check whether the edge between two coordinates exists.

        Args:
            a: (x, y) of first node.
            b: (x, y) of second node.

        Returns:
            True if a<->b exists, else False.
        """
        ia, ib = self.index_of(a), self.index_of(b)
        return (ib in self.adj[ia]) and (ia in self.adj[ib])

    def remove_edge(self, a: Coord, b: Coord):
        """
        Remove the edge between two coordinates if the edge exists

        Args:
            a: (x, y) of first node.
            b: (x, y) of second node.
        """
        ia, ib = self.index_of(a), self.index_of(b)
        self.adj[ia].discard(ib)
        self.adj[ib].discard(ia)

    def edges(self) -> Iterable[Tuple[Coord, Coord]]:
        """
        Iterate all unique edges in coordinate space.

        Yields:
            Tuples (coord_a, coord_b) for each edge once.
        """
        seen = set()
        for a_idx, nbrs in self.adj.items():
            for b_idx in nbrs:
                key = tuple(sorted([a_idx, b_idx]))
                if key not in seen:
                    seen.add(key)
                    yield (self.coord_of(a_idx), self.coord_of(b_idx))

    def _compute_normalizer(self, extra_points: Optional[List[Coord]] = None,
                            img_size=700, margin=50):
        """
        Prepare a coordinate->pixel mapping for visualization.

        Args:
            extra_points: optional extra (x, y) points to include in bounds.
            img_size: output square image size in pixels.
            margin: padding in pixels.

        Returns:
            (to_px, img_size, margin) where to_px(x, y)->(u, v) pixel function.
        """
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        if extra_points:
            xs.extend([p[0] for p in extra_points])
            ys.extend([p[1] for p in extra_points])
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad_x = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        pad_y = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        xmin -= pad_x; xmax += pad_x
        ymin -= pad_y; ymax += pad_y
        xr = xmax - xmin if xmax > xmin else 1.0
        yr = ymax - ymin if ymax > ymin else 1.0
        S = img_size - 2 * margin
        def to_px(x, y):
            nx = (x - xmin) / xr
            ny = (y - ymin) / yr
            nx = 0.0 if nx < 0.0 else (1.0 if nx > 1.0 else nx)
            ny = 0.0 if ny < 0.0 else (1.0 if ny > 1.0 else ny)
            u = margin + int(nx * S)
            v = margin + int((1.0 - ny) * S)
            return (u, v)
        return to_px, img_size, margin

    def base_image(self, extra_points: Optional[List[Coord]] = None,
                   img_size=700, margin=50) -> Tuple[np.ndarray, callable]:
        """
        Create a blank white image and a to_px mapping for drawing the graph.

        Args:
            extra_points: optional (x, y) points to include in normalization.
            img_size: output image size in pixels.
            margin: border margin in pixels.

        Returns:
            (img, to_px) where img is an (H x H x 3) uint8 canvas,
            and to_px is a function mapping (x, y)->(u, v).
        """
        to_px, H, _ = self._compute_normalizer(extra_points, img_size, margin)
        img = np.ones((H, H, 3), dtype=np.uint8) * 255
        return img, to_px

# -------- path util ----------
def nearest_point_on_polyline(path: List[Coord], p: Coord) -> Tuple[float, int, Coord]:
    """
    Project a point onto a polyline and find the closest point/segment.

    Args:
        path: list of waypoints [(x, y), ...].
        p: query point (x, y).

    Returns:
        (s, seg_idx, proj) where:
          - s is arclength from path[0] to the projection point (meters),
          - seg_idx is the index of the segment [path[i], path[i+1]] containing proj,
          - proj is the projected (x, y) point on the path.
    """
    if len(path) == 1:
        return 0.0, 0, path[0]
    def dot(a, b): return a[0]*b[0] + a[1]*b[1]
    def sub(a, b): return (a[0]-b[0], a[1]-b[1])
    def add(a, b): return (a[0]+b[0], a[1]+b[1])
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + math.dist(path[i-1], path[i]))
    best_s = 0.0; best_pt = path[0]; best_seg = 0
    for i in range(1, len(path)):
        a = path[i-1]; b = path[i]
        ab = sub(b, a); ap = sub(p, a)
        ab2 = dot(ab, ab) if ab != (0.0, 0.0) else 1e-9
        t = max(0.0, min(1.0, dot(ap, ab) / ab2))
        proj = add(a, (ab[0]*t, ab[1]*t))
        s_here = cum[i-1] + math.dist(a, proj)
        if i == 1 or math.dist(p, proj) < math.dist(p, best_pt):
            best_s = s_here; best_pt = proj; best_seg = i - 1
    return best_s, best_seg, best_pt

def lookahead_point(path: List[Coord], p: Coord, Ld: float) -> Coord:
    """
    Find the point at arclength along a polyline from the projection of p.

    Args:
        path: list of waypoints [(x, y), ...].
        p: current position (x, y).
        Ld: lookahead distance in meters.

    Returns:
        (x, y) of the lookahead point clamped to the end of the path.
    """
    if not path: return p
    if len(path) == 1: return path[0]
    s, _, _ = nearest_point_on_polyline(path, p)
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + math.dist(path[i-1], path[i]))
    target_s = s + Ld
    if target_s >= cum[-1]: return path[-1]
    for i in range(1, len(path)):
        if cum[i] >= target_s:
            seg_s = target_s - cum[i-1]
            a = path[i-1]; b = path[i]
            seg_len = math.dist(a, b)
            t = seg_s / seg_len
            return (a[0] + (b[0]-a[0]) * t, a[1] + (b[1]-a[1]) * t)
    return path[-1]

def angle_wrap(a: float) -> float:
    """
    Wrap an angle to the range [-pi, pi].

    Args:
        a: angle in radians.

    Returns:
        Wrapped angle in radians.
    """
    while a > math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a

def yaw_from_quat(x, y, z, w) -> float:
    """
    Extract yaw (heading about +Z) from a quaternion.

    Args:
        x, y, z, w: quaternion components.

    Returns:
        Yaw in radians in the odometry frame
    """
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

# --------- Simple PID ----------
@dataclass
class PID:
    """
    Minimal PID controller with anti-windup (integrator clamping).
    Call update(e, dt) each cycle to get the control effort.
    """
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    i: float = 0.0
    prev_e: Optional[float] = None
    i_limit: float = 1.0

    def reset(self):
        """
        Reset internal integrator and derivative memory.
        """
        self.i = 0.0
        self.prev_e = None

    def update(self, e: float, dt: float) -> float:
        """
        Compute PID output for a given error and timestep.

        Args:
            e: control error (setpoint - measurement).
            dt: time step in seconds.

        Returns:
            Control effort (float).
        """
        if dt <= 0.0:
            dterm = 0.0
        else:
            dterm = 0.0 if self.prev_e is None else (e - self.prev_e) / dt
        self.i += e * dt
        self.i = float(np.clip(self.i, -self.i_limit, self.i_limit))
        self.prev_e = e
        return self.kp * e + self.ki * self.i + self.kd * dterm

# --------------- Node ---------------
class GridPathFollowerNode(Node):
    """
    ROS2 node that:
    - Plans on a grid using A* between start/goal nodes,
    - Follows the path with a suitable control mechanism,
    - Backtracks and replans if an obstacle is not passable
    """
    def __init__(self):
        """
        Initialize parameters, build graph, subscribe/publish ROS topics,
        and compute the initial path.
        """
        super().__init__("grid_path_follower")

        # --- Motion and geometry parameters ---
        self.goal_tol = 0.05
        self.node_tol = 0.05
        # TODO: YOUR CODE HERE: ~3 lines: set the lookahead distance for pure-pursuit based control, and the maximum allowed linear and angular velocities 
        self.lookahead_dist = 0.35
        self.v_max = 0.40
        self.w_max = 2.40
        self.robot_radius = 0.089
        self.safety_margin = 0.02
        self.grid_res = 0.025
        self.plan_window = 0.95
        self.plan_half_width = 0.45
        self.narrow_speed = 0.12
        self.node_speed = 0.18
        # ...
        # TODO: YOUR CODE HERE: ~1-2 lines: set your PID controller/s for linear/angular motion
        self.pid_angular  = PID(kp=2.2, ki=0.0, kd=0.10, i_limit=0.5)

        # --- Helpers to halt turtlebot ---
        self.dwell_s = 0.5
        self.dwell_until: Optional[float] = None

        # --- Start and goal positions. TODO: OPTIONAL: modify for testing purposes as needed ---
        self.start_pos = (0.0, 0.0)
        self.goal_pos  = (7.120, -5.696)

        # --- State ---
        self.pose_ready = False                    # to check if odometry values are being received
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0 # turtlebot pose 
        self.scan: Optional[LaserScan] = None      # stores the latest scan
        self.mode = 'None'                         # operation mode 
        self.last_t = self._now()

        # --- Grid (5x5) ---
        self.node_graph = [
            [[0.712, 0], [2.136, 0], [3.560, 0], [4.984, 0], [6.408, 0]],
            [[0.712, -1.424], [2.136, -1.424], [3.560, -1.424], [4.984, -1.424], [6.408, -1.424]],
            [[0.712, -2.848], [2.136, -2.848], [3.560, -2.848], [4.984, -2.848], [6.408, -2.848]],
            [[0.712, -4.272], [2.136, -4.272], [3.560, -4.272], [4.984, -4.272], [6.408, -4.272]],
            [[0.712, -5.696], [2.136, -5.696], [3.560, -5.696], [4.984, -5.696], [6.408, -5.696]],
        ]
        # Graph and initial plan
        self.graph = GridGraph.from_node_graph(self.node_graph)
        """
        GridGraph.from_node_graph returns:
                 GridGraph(
                    H=5, 
                    W=5, 
                    positions={(0, 0): (0.712, 0.0), (0, 1): (2.136, 0.0), ....}
                    coord2idx={(0.712, 0.0): (0, 0), (2.136, 0.0): (0, 1), ....}
                    adj={(0, 0): {(1, 0), (0, 1)}, (0, 1): {(0, 2), (1, 1), (0, 0)}, (0, 2): {(0, 1), (1, 2), (0, 3)}, ........., (4, 4): {(3, 4), (4, 3)}}
                   )
        """
        self.full_path:  List[Coord] = []
        self.prev_node : Optional[Coord] = None
        self.block_edge: Optional[Tuple[Coord, Coord]] = None
        self.blocked_counts: Dict[Tuple[Coord, Coord], int] = {}
        self._plan_cache = None
        self.speed_cap = float('inf')
        self.pose_hist = []
        self.blocked_votes = 0
        self.blocked_votes_needed = 3
        self.yaw_corr = 0.0
        #TODO: OPTIONAL: you can manually remove edges here using 'self.graph.remove_edge([..., ...], [..., ...]))

        # --- ROS I/O ---
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 20)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 20)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 20)
        self.pose_sub = self.create_subscription(Odometry, "/odom", self.pose_hist_cb, 20)
        self.yaw_est_sub = self.create_subscription(LaserScan, "/scan", self.yaw_est_cb, 20)
        self.timer = self.create_timer(1.0/20.0, self.control_step)

        # --- Modes of operation ---
        """
            1. FOLLOW: when in default mode of following the A* generated path  
            2. BACKTRACK: when an unpassable obstacle is encountered (go back to immediately previous node) 
            3. REPLAN: recalculate A* to get new path to the goal from the current node
            4. DONE: when goal has been reached
        """
        self._set_mode('FOLLOW') 

        # --- Vizualization ---
        self.win = "Grid A* Path Follower"
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)

        # --- Initial plan ----
        self.path_plan(self.start_pos)
        self.get_logger().info(f"Planned {len(self.full_path)} waypoints.")

    # --- OPTIONAL helper: time ---
    def _now(self) -> float:
        """
        Get ROS2 clock time in seconds (float).

        Returns:
            Current time in seconds as a float.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    # --- Helper to set the mode of operation ---
    def _set_mode(self, new_mode: str):
        """
        Switch high-level mode and enforce a short dwell (pause).

        Args:
            new_mode: one of {'FOLLOW', 'BACKTRACK', 'REPLAN', 'DONE'}.
        """
        if self.mode != new_mode:
            self.mode = new_mode
            self.publish_stop()
            self.dwell_until = self._now() + self.dwell_s
            self.get_logger().info(f"Mode -> {self.mode}, pausing {self.dwell_s}s")

    # --- planning ---
    # --- A* implementation ---
    def astar_path(self, graph: GridGraph, start: Coord, goal: Coord) -> List[Coord]:
        """
        Run A* on the GridGraph between two coordinates.

        Args:
            graph: GridGraph instance.
            start: (x, y) start coordinate
            goal:  (x, y) goal coordinate 

        Returns:
            List of (x, y) coordinates along the path from start node to goal node,
            inclusive. Empty list if no path.

        Hint: use the relevant GriGraph functions)
        """
        start_idx = graph.index_of(start)
        goal_idx = graph.index_of(goal)
        def h(i: Index) -> float:
            x1, y1 = graph.coord_of(i); x2, y2 = graph.coord_of(goal_idx)
            return math.hypot(x2 - x1, y2 - y1)
        def c(i: Index, j: Index) -> float:
            x1, y1 = graph.coord_of(i); x2, y2 = graph.coord_of(j)
            return math.hypot(x2 - x1, y2 - y1)
        open_heap = [(h(start_idx), start_idx)]
        came: Dict[Index, Index] = {}
        g = {start_idx: 0.0}
        in_open = {start_idx}
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            in_open.discard(cur)
            if cur == goal_idx:
                path_idx = [cur]
                while cur in came:
                    cur = came[cur]
                    path_idx.append(cur)
                path_idx.reverse()
                return [graph.coord_of(i) for i in path_idx]
            for nbr in graph.adj[cur]:
                t = g[cur] + c(cur, nbr)
                if t < g.get(nbr, float('inf')):
                    came[nbr] = cur
                    g[nbr] = t
                    f = t + h(nbr)
                    if nbr not in in_open:
                        heapq.heappush(open_heap, (f, nbr))
                        in_open.add(nbr)
        return []

    def nearest_graph_coord(self, p: Coord) -> Coord:
        """
        Find the nearest graph node (in metric space) to a given point.

        Args:
            p: (x, y) query position.

        Returns:
            (x, y) of the closest node in the GridGraph.
        """
        best = None; best_d = float('inf')
        for xy in self.graph.coord2idx.keys():
            d = math.dist(p, xy)
            if d < best_d: best_d = d; best = xy
        return best

    def _is_graph_node(self, p: Coord) -> bool:
        try:
            self.graph.index_of(p)
            return True
        except KeyError:
            return False

    def _restore_blocked_edges(self, max_count: Optional[int]):
        self.graph = GridGraph.from_node_graph(self.node_graph)
        for (a, b), count in self.blocked_counts.items():
            if max_count is not None and count > max_count:
                self.graph.remove_edge(a, b)
        self.get_logger().warn("No route found: restoring blocked edges")

    # ---  Invoke A* to return the nodes that define path ---
    def plan_from(self, start_pose: Coord) -> List[Coord]:
        """
        Build a waypoint list: start_pose -> nearest start node -> A* nodes
        -> nearest goal node -> goal.

        Args:
            start_pose: (x, y) where to start planning from.

        Returns:
            List of (x, y) waypoints including start_pose and goal_pos.
        """
        # TODO: YOUR CODE HERE: ~ 3 lines
        # 1. Get the closest graph node to the robot’s current starting position (ns)
        # 2. Get the closest graph node to the goal position (ng)
        # 3. Run A* to get a path (sequence of graph nodes) from ns → ng
        ns = self.nearest_graph_coord(start_pose)
        ng = self.nearest_graph_coord(self.goal_pos)
        a_star_nodes = self.astar_path(self.graph, ns, ng)

        if not a_star_nodes:
            self._restore_blocked_edges(max_count=1)
            a_star_nodes = self.astar_path(self.graph, ns, ng)
        if not a_star_nodes:
            self._restore_blocked_edges(max_count=None)
            a_star_nodes = self.astar_path(self.graph, ns, ng)

        path: List[Coord] = [start_pose]

        if not a_star_nodes:
            self.get_logger().error(f"A* found no path from {ns} to {ng}; stopping.")
            return path

        # TODO: YOUR CODE HERE: ~8 lines
        # 5. If the start node is different from the current robot position, append it to 'path'
        # 6. Append all waypoints from the A* grid path (avoiding duplicate points)
        # 7. Ensure the final node (ng) is included
        # 8. Ensure the actual goal position is included as the last waypoint
        if math.dist(start_pose, ns) > 0.05:
            path.append(ns)
        for pt in a_star_nodes:
            if math.dist(path[-1], pt) > 1e-4:
                path.append(pt)
        if math.dist(path[-1], ng) > 1e-4:
            path.append(ng)
        if math.dist(path[-1], self.goal_pos) > 1e-4:
            path.append(self.goal_pos)

        # Return the full planned path as a list of coordinates
        return path

    def path_plan(self, start_from: Optional[Coord] = None):
        """
        (Re)plan a global path and store it in self.full_path.

        Args:
            start_from: optional (x, y) to start planning from; if None,
                        uses current odom pose (if available) else start_pos.
        """
        if start_from is None:
            start_from = (self.x, self.y) if self.pose_ready else self.start_pos
        self.full_path = self.plan_from(start_from)
        self.get_logger().info(f"Path planned from {start_from} with {len(self.full_path)} waypoints.")

    # --- callbacks ---
    def odom_cb(self, msg: Odometry):
        """
        Odometry callback. Updates pose estimate (x, y, yaw).

        Args:
            msg: nav_msgs/Odometry message.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.pose_ready = True

    def scan_cb(self, msg: LaserScan):
        """
        LaserScan callback. Stores the latest scan.

        Args:
            msg: sensor_msgs/LaserScan message.
        """
        self.scan = msg

    # --- TODO: YOUR CODE HERE: Implement the function to determine passability  ---
    def is_passable(self, seg_dir_yaw: float) -> bool:
        """
        Decide if the edge between two nodes is passable or not

        Args:
            seg_dir_yaw: segment heading in world/odom frame (radians).

        Returns:
            True if passable; False otherwise.
        """
        if self.scan is None:
            return True

        if self.mode == 'DONE':
            return True

        plan = self._local_plan(seg_dir_yaw)
        if plan is None:
            return True
        if plan[0]:
            self.blocked_votes = 0
            return True

        self.blocked_votes += 1
        if self.blocked_votes < self.blocked_votes_needed:
            return True
        self.blocked_votes = 0
        self.get_logger().info(f"Segment blocked at ({self.x:.2f}, {self.y:.2f})")
        return False

    def pose_hist_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self.pose_hist and t < self.pose_hist[-1][0] - 0.5:
            self.pose_hist.clear()
        self.pose_hist.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y,
                               yaw_from_quat(q.x, q.y, q.z, q.w)))
        while len(self.pose_hist) > 1 and t - self.pose_hist[0][0] > 2.0:
            self.pose_hist.pop(0)

    def yaw_est_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        ok = np.isfinite(ranges) & (ranges > 0.25) & (ranges < 0.8)
        pts = np.column_stack((np.where(ok, ranges, 0.0) * np.cos(angles), np.where(ok, ranges, 0.0) * np.sin(angles)))
        stamp = msg.header.stamp
        yaw = self.pose_at(stamp.sec + 1e-9 * stamp.nanosec)[2] - self.yaw_corr

        resid = []
        for k in range(0, len(ranges) - 14, 7):
            if not ok[k:k + 14].all():
                continue
            piece = pts[k:k + 14] - pts[k:k + 14].mean(axis=0)
            _, vecs = np.linalg.eigh(piece.T @ piece)
            if abs(piece @ vecs[:, 0]).max() > 0.015:
                continue
            heading = math.atan2(vecs[1, 1], vecs[0, 1]) + yaw
            resid.append((heading + math.pi / 4) % (math.pi / 2) - math.pi / 4)
        resid = np.array(resid)
        close = np.abs(resid) < 0.2
        if np.count_nonzero(close) < 3:
            return
        self.yaw_corr += 0.3 * float(resid[close].mean())

    def pose_at(self, t: float) -> Tuple[float, float, float]:
        h = self.pose_hist
        if len(h) < 2 or t < h[0][0] or t > h[-1][0]:
            return (self.x, self.y, self.yaw)
        for k in range(1, len(h)):
            if h[k][0] >= t:
                t0, x0, y0, a0 = h[k - 1]
                t1, x1, y1, a1 = h[k]
                f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                return (x0 + f * (x1 - x0), y0 + f * (y1 - y0), angle_wrap(a0 + f * angle_wrap(a1 - a0)))
        return (self.x, self.y, self.yaw)

    def _local_plan(self, seg_dir_yaw: float):
        key = (id(self.scan), self.x, self.y, seg_dir_yaw)
        if self._plan_cache is not None and self._plan_cache[0] == key:
            return self._plan_cache[1]
        result = self._compute_local_plan(seg_dir_yaw)
        self._plan_cache = (key, result)
        return result

    def _compute_local_plan(self, seg_dir_yaw: float):
        ranges = np.array(self.scan.ranges)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment
        valid = (ranges >= 0.08) & (ranges <= 1.3) & np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        stamp = self.scan.header.stamp
        t_scan = stamp.sec + 1e-9 * stamp.nanosec
        scan_x, scan_y, scan_yaw = self.pose_at(t_scan)
        scan_yaw -= self.yaw_corr
        cos_y, sin_y = math.cos(scan_yaw), math.sin(scan_yaw)
        px = ranges * np.cos(angles)
        py = ranges * np.sin(angles)
        world_x = scan_x + px * cos_y - py * sin_y
        world_y = scan_y + px * sin_y + py * cos_y

        _, seg_idx, _ = nearest_point_on_polyline(self.full_path, (self.x, self.y))
        ga = self.full_path[seg_idx]
        gb = self.full_path[seg_idx + 1]

        cos_s, sin_s = math.cos(seg_dir_yaw), math.sin(seg_dir_yaw)
        pts_s = (world_x - ga[0]) * cos_s + (world_y - ga[1]) * sin_s
        pts_l = -(world_x - ga[0]) * sin_s + (world_y - ga[1]) * cos_s
        rob_s = (self.x - ga[0]) * cos_s + (self.y - ga[1]) * sin_s
        rob_l = -(self.x - ga[0]) * sin_s + (self.y - ga[1]) * cos_s
        seg_length = math.dist(ga, gb)

        s_near = rob_s
        s_far = min(rob_s + self.plan_window, seg_length - 0.05)
        if s_far - s_near < 0.10:
            return None

        res = self.grid_res
        half_w = self.plan_half_width
        ns = int((s_far - s_near) / res)
        nl = int(2 * half_w / res)
        cell_s = s_near + (np.arange(ns) + 0.5) * res
        cell_l = -half_w + (np.arange(nl) + 0.5) * res

        near = (pts_s > s_near - 0.3) & (pts_s < s_far + 0.3) & (np.abs(pts_l) < 0.75)
        ps, pl = pts_s[near], pts_l[near]
        if len(ps) > 0:
            gs, gl = np.meshgrid(cell_s, cell_l, indexing='ij')
            dist = np.sqrt((gs[..., None] - ps) ** 2 + (gl[..., None] - pl) ** 2).min(axis=2)
        else:
            dist = np.full((ns, nl), 1.0)

        free = dist > self.robot_radius + self.safety_margin
        j0 = int(np.clip((rob_l + half_w) / res, 0, nl - 1))
        free[:2, max(0, j0 - 3):j0 + 4] = True
        penalty = 8.0 * np.clip((0.22 - dist) / 0.22, 0.0, 1.0)

        cost = np.full((ns, nl), np.inf)
        parent = {}
        cost[0, j0] = 0.0
        heap = [(0.0, 0, j0)]
        moves = [(1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414), (0, 1, 1.0), (0, -1, 1.0)]
        while heap:
            c, i, j = heapq.heappop(heap)
            if c > cost[i, j]:
                continue
            for di, dj, step in moves:
                ni, nj = i + di, j + dj
                if 0 <= ni < ns and 0 <= nj < nl and free[ni, nj]:
                    nc = c + step * res * (1.0 + penalty[ni, nj])
                    if nc < cost[ni, nj]:
                        cost[ni, nj] = nc
                        parent[(ni, nj)] = (i, j)
                        heapq.heappush(heap, (nc, ni, nj))

        pull = 20.0 if s_far > seg_length - 0.06 else 0.5
        end_cost = cost[ns - 1] + pull * np.abs(cell_l)
        j_end = int(np.argmin(end_cost))
        if not np.isfinite(end_cost[j_end]):
            return (False, None, float('inf'))

        cells = [(ns - 1, j_end)]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        cells.reverse()
        path = []
        for i, j in cells:
            path.append((ga[0] + cell_s[i] * cos_s - cell_l[j] * sin_s,
                         ga[1] + cell_s[i] * sin_s + cell_l[j] * cos_s))
        speed_cap = float('inf')
        if min(dist[i, j] for i, j in cells[:24]) < 0.19:
            speed_cap = self.narrow_speed
        elif len(cells) <= 12:
            speed_cap = self.node_speed
        return (True, path, speed_cap)

    # --- TODO: YOUR CODE HERE: Implement the function to generate the lookahead point away from obstacles  ---
    def obstacle_avoided_target(self, robot_pos: Coord, seg_dir_yaw: float) -> Coord:
        """
        Generate a target point that accounts for obstacle avoidance:
        - Pure-pursuit base target at lookahead distance along the path.
        - Lateral shift inside the road corridor to increase clearance.

        Args:
            robot_pos: current turtlebot position (x, y) in odom/world frame.
            seg_dir_yaw: segment heading in world/odom frame (radians).

        Returns:
            Target (x, y) in world/odom frame.
        """
        if self.scan is None:
            return lookahead_point(self.full_path, robot_pos, self.lookahead_dist)

        _, seg_idx, _ = nearest_point_on_polyline(self.full_path, robot_pos)
        ga = self.full_path[seg_idx]
        gb = self.full_path[seg_idx + 1]
        passed = (robot_pos[0] - ga[0]) * math.cos(seg_dir_yaw) + (robot_pos[1] - ga[1]) * math.sin(seg_dir_yaw)
        if passed > math.dist(ga, gb) + 0.03:
            return gb

        plan = self._local_plan(seg_dir_yaw)
        if plan is not None and plan[0]:
            self.speed_cap = plan[2]
            look = 0.22 if self.speed_cap <= self.narrow_speed else self.lookahead_dist
            return lookahead_point([robot_pos] + plan[1], robot_pos, look)
        self.speed_cap = float('inf')

        ranges = np.array(self.scan.ranges)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment
        valid = (ranges >= 0.08) & (ranges <= 1.2) & np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        px = ranges * np.cos(angles)
        py = ranges * np.sin(angles)

        cos_y, sin_y = math.cos(self.yaw - self.yaw_corr), math.sin(self.yaw - self.yaw_corr)
        world_x = self.x + px * cos_y - py * sin_y
        world_y = self.y + px * sin_y + py * cos_y

        dx = world_x - ga[0]
        dy = world_y - ga[1]
        cos_s, sin_s = math.cos(seg_dir_yaw), math.sin(seg_dir_yaw)
        long_dist = dx * cos_s + dy * sin_s
        lat_dist = -dx * sin_s + dy * cos_s

        rob_dx = robot_pos[0] - ga[0]
        rob_dy = robot_pos[1] - ga[1]
        rob_s = rob_dx * cos_s + rob_dy * sin_s

        near_mask = (long_dist > rob_s + 0.08) & (long_dist < rob_s + 0.45) & (np.abs(lat_dist) < 0.25)
        near_lat = lat_dist[near_mask]

        target_lat_offset = 0.0

        if len(near_lat) > 0:
            left_pts = near_lat[near_lat > 0.03]
            right_pts = near_lat[near_lat < -0.03]

            if len(left_pts) >= 3 and len(right_pts) >= 3:
                target_lat_offset = float(np.clip(0.5 * (np.min(left_pts) + np.max(right_pts)), -0.12, 0.12))
            elif len(left_pts) >= 3:
                target_lat_offset = -0.12
            elif len(right_pts) >= 3:
                target_lat_offset = 0.12

        eff_Ld = 0.24 if abs(target_lat_offset) > 0.04 else self.lookahead_dist
        tgt_s = rob_s + eff_Ld

        if seg_idx + 1 == len(self.full_path) - 1 and tgt_s >= math.dist(ga, gb):
            return gb

        tgt_x = ga[0] + tgt_s * cos_s - target_lat_offset * sin_s
        tgt_y = ga[1] + tgt_s * sin_s + target_lat_offset * cos_s

        return (tgt_x, tgt_y)
    
    # --- TODO: YOUR CODE HERE: controller used for TurtleBot ---
    def turtlebot_control(self, robot_pos: Coord, robot_h: float, target: Coord, dt: float, Ld: float) -> Tuple[float, float]:
        """
        Compute (v, w) commands that blend pure-pursuit with PID:

        Args:
            robot_pos: current pose (x, y) in odom/world frame.
            robot_h: current heading (yaw angle) in the odom/world frame.
            target: target point (x, y) to track.
            dt: timestep (s).
            Ld: pure-pursuit lookahead distance (m).

        Returns:
            (v_cmd, w_cmd): linear and angular velocity commands.
        """
        if self.mode == 'DONE':
            return 0.0, 0.0

        robot_h -= self.yaw_corr

        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        dist_to_tgt = math.hypot(dx, dy)

        if dist_to_tgt < 1e-3:
            return 0.0, 0.0

        cos_h = math.cos(robot_h)
        sin_h = math.sin(robot_h)
        x_L = dx * cos_h + dy * sin_h
        y_L = -dx * sin_h + dy * cos_h

        alpha = math.atan2(y_L, x_L)

        repulsion_bias = 0.0
        front_collision = False

        if self.scan is not None:
            ranges = np.array(self.scan.ranges)
            angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment
            valid = (ranges >= 0.05) & (ranges <= 0.35) & np.isfinite(ranges)
            
            if np.any(valid):
                px = ranges[valid] * np.cos(angles[valid])
                py = ranges[valid] * np.sin(angles[valid])

                near_side = (px > -0.05) & (px < 0.22) & (np.abs(py) < 0.14)
                if np.any(near_side):
                    py_near = py[near_side]
                    left_close = py_near[py_near > 0.0]
                    right_close = py_near[py_near < 0.0]

                    min_left = np.min(left_close) if len(left_close) > 0 else 0.18
                    min_right = np.abs(np.max(right_close)) if len(right_close) > 0 else 0.18

                    if min_left < 0.13:
                        repulsion_bias -= 1.8 * (0.13 - min_left)
                    if min_right < 0.13:
                        repulsion_bias += 1.8 * (0.13 - min_right)

                front_pts = (px > 0.04) & (px < 0.10) & (np.abs(py) < 0.05)
                if np.any(front_pts):
                    front_collision = True

        alpha_eff = angle_wrap(alpha + repulsion_bias)

        if abs(alpha_eff) > 0.45 or front_collision:
            w_cmd = float(np.clip(2.8 * alpha_eff, -self.w_max, self.w_max))
            return 0.0, w_cmd

        curvature = (2.0 * math.sin(alpha_eff)) / max(dist_to_tgt, 0.16)

        v_base = 0.22 if abs(repulsion_bias) > 0.04 else self.v_max
        v_cmd = max(0.18, v_base * max(0.40, math.cos(alpha_eff)))
        v_cmd = min(v_cmd, self.speed_cap)
        if math.dist(robot_pos, self.goal_pos) < 0.6:
            v_cmd = min(v_cmd, max(0.06, 1.2 * dist_to_tgt))

        w_cmd = (v_cmd * curvature) + self.pid_angular.update(alpha_eff, dt)
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))
        v_cmd = float(np.clip(v_cmd, 0.0, self.v_max))

        return v_cmd, w_cmd

    # --- Publisher helpers ---
    def publish_cmd(self, v: float, w: float):
        """
        Publish a geometry_msgs/Twist command.

        Args:
            v: linear velocity (m/s), applied to msg.linear.x.
            w: angular velocity (rad/s), applied to msg.angular.z.
        """
        msg = Twist(); msg.linear.x = float(v); msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        """
        Convenience: publish zero velocities to stop the turtlebot.
        """
        self.publish_cmd(0.0, 0.0)

    # --- Vizualization helper ---
    def draw_scene(self, robot: Optional[Coord] = None, target: Optional[Coord] = None):
        """
        Render the grid, nodes, path, turtlebot and target for debugging.

        Args:
            robot: optional (x, y) turtlebot position to plot.
            target: optional (x, y) target point to plot.
        """
       
        extra = [self.start_pos, self.goal_pos]
        if robot: extra.append(robot)
        if target: extra.append(target)
        img, to_px = self.graph.base_image(extra_points=extra, img_size=800, margin=60)
        # edges (grey)
        for a, b in self.graph.edges():
            cv2.line(img, to_px(*a), to_px(*b), (220, 220, 220), 2)
        # nodes (purple)
        for xy in self.graph.coord2idx.keys():
            cv2.circle(img, to_px(*xy), 6, (240, 32, 160), -1)
        # path (black)
        if self.full_path and len(self.full_path) >= 2:
            pts = np.array([to_px(*p) for p in self.full_path], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(30, 30, 30), thickness=3)

        # start (red) & goal (green)
        cv2.circle(img, to_px(*self.start_pos), 9, (0, 0, 220), -1)
        cv2.circle(img, to_px(*self.goal_pos), 9, (0, 220, 0), -1)
        # turtlebot
        if robot is not None:
            pr = to_px(*robot)
            cv2.circle(img, pr, 7, (0, 165, 255), -1)
            hx = robot[0] + math.cos(self.yaw) * 0.3
            hy = robot[1] + math.sin(self.yaw) * 0.3
            cv2.arrowedLine(img, pr, to_px(hx, hy), (0, 165, 255), 2, tipLength=0.3)
        # lookahead target (blue)
        if target is not None:
            cv2.circle(img, to_px(*target), 7, (200, 0, 0), -1) 
        # HUD
        cv2.putText(img, f"Mode: {self.mode}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow(self.win, img); cv2.waitKey(1)
    
    # --- Main control loop ----
    def control_step(self):
        """
        Main control loop
        """

        # --- Stamping current time ---
        now = self._now()
        dt = max(1e-3, now - self.last_t)
        self.last_t = now

        # --- No odometry values, no calculated A* path, or path is only one coordinate ---
        if not self.pose_ready or self.full_path is None or len(self.full_path) < 2:
            self.draw_scene()
            return

        # --- Current turtlebot position ---
        robot_position = (self.x, self.y)
        robot_heading  = self.yaw

        # --- Check if goal reached ---
        if math.dist(robot_position, self.goal_pos) < self.goal_tol:
            self.publish_stop()
            self._set_mode('DONE')
            self.get_logger().info(f'GOAL REACHED!!!')
            self.draw_scene(robot=robot_position)
            return

        # --- (Optional) Pause motion ---
        if self.dwell_until is not None and now < self.dwell_until:
            self.publish_stop()
            self.draw_scene(robot=robot_position)
            return
        else:
            self.dwell_until = None

        # --- Find the current path segment and its properties ---
        _, seg_idx, _ = nearest_point_on_polyline(self.full_path, robot_position) # which path segment (between two waypoints) the turtlebot is currently on
        ga = self.full_path[seg_idx]     # segment’s start point
        gb = self.full_path[seg_idx + 1] # segment’s end point
        seg_vec = (gb[0] - ga[0], gb[1] - ga[1]) # vector pointing from a → b (i.e. local forward direction)
        seg_dir_yaw = math.atan2(seg_vec[1], seg_vec[0]) # heading (yaw angle) of this segment in the world frame
        dist_to_b = math.dist(robot_position, gb) # robot’s straight-line distance to the next waypoint (b)

        # --- Check arrival at the node 'b' and (Optional) pause motion ---
        if dist_to_b <= self.node_tol:
            if seg_idx + 1 < len(self.full_path):
                self.full_path = self.full_path[seg_idx + 1:]
            self.publish_stop()
            self.dwell_until = self._now() + self.dwell_s
            self.draw_scene(robot=robot_position, target=gb)
            return

        # --- MODE: REPLAN ---
        if self.mode == 'REPLAN':
            self.publish_stop()
            # TODO: YOUR CODE HERE: 
            # ~2 lines: Remove the edge if the edge is blocked 
            # ~1 line:  Set start_from to the previous node
            # ~1 line:  Replan the global path from new start point and store it in self.full_path.
            # ~1 line:  Set the mode back to FOLLOW
            if self.block_edge is not None:
                a, b = self.block_edge
                if self._is_graph_node(a) and self._is_graph_node(b):
                    self.graph.remove_edge(a, b)
                    key = tuple(sorted([a, b]))
                    self.blocked_counts[key] = self.blocked_counts.get(key, 0) + 1
                else:
                    self.get_logger().warn(f"Blocked segment {a} -> {b} is not a graph edge; nothing to remove.")
                self.block_edge = None

            start_from = self.prev_node if self.prev_node is not None else self.nearest_graph_coord(robot_position)
            self.path_plan(start_from)
            self._set_mode('FOLLOW')
            self.draw_scene(robot=robot_position)
            return

        # --- MODE: BACKTRACK ---
        if self.mode == 'BACKTRACK':
            # TODO: YOUR CODE HERE
            # ~1 line: set a temporary goal as the previous node
            temp_goal = self.prev_node if self.prev_node is not None else ga

            back_path = [robot_position, temp_goal]
            target = lookahead_point(back_path, robot_position, self.lookahead_dist)
            v_cmd, w_cmd = self.turtlebot_control(robot_position, robot_heading, target, dt, self.lookahead_dist) # Get linear and angular velocity values from your control block
            self.publish_cmd(v_cmd, w_cmd) # publish the velocities
            self.draw_scene(robot=robot_position, target=target)

            # TODO: YOUR CODE HERE: ~3 lines
            # If the distance between the current turtlebot position ('robot') and temp_goal is less than the tolerance ('self.node_tol'), stop the turtlebot immediately and set the mode to REPLAN
            if math.dist(robot_position, temp_goal) < self.node_tol:
                self.publish_stop()
                self._set_mode('REPLAN')
            return

        # --- MODE: FOLLOW ---
        # React if an obstacle has become non-passable
        if not self.is_passable(seg_dir_yaw):
            # TODO: YOUR CODE HERE: ~4 lines
            # 1. Stop the turtlebot immediately
            # 2. set 'prev_node' to 'ga' 
            # 3. set 'block_edge' to '(ga, gb)' 
            # 4. set mode to 'BACKTRACK'
            self.publish_stop()
            self.prev_node = ga
            self.block_edge = (ga, gb)
            self._set_mode('BACKTRACK')
            self.draw_scene(robot=robot_position)
            return
            
        target = self.obstacle_avoided_target(robot_position, seg_dir_yaw) # Get obstacle avoided lookahead target point
        v_cmd, w_cmd = self.turtlebot_control(robot_position, robot_heading, target, dt, self.lookahead_dist) # Get linear and angular velocity values from your control block
        self.publish_cmd(v_cmd, w_cmd) # publish the velocities
        self.draw_scene(robot=robot_position, target=target)


# --------------- entry ---------------
def main():
    rclpy.init()
    node = GridPathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

