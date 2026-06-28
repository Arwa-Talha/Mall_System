"""
parking_graph.py  —  column-aware graph + A* with road waypoints
Road waypoints sit in the aisles BETWEEN columns so the path
follows the actual driving lanes, not cuts through parked cars.
"""

import json, math, heapq
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpotNode:
    id: int
    cx: float
    cy: float
    points: list
    status: str = "unknown"
    is_waypoint: bool = False   # True = road waypoint, not a parking spot


@dataclass(order=True)
class _PQItem:
    priority: float
    node_id: int = field(compare=False)


class ParkingGraph:
    def __init__(self, json_path: str, col_threshold: int = 80):
        self.spots: dict[int, SpotNode] = {}
        self.adjacency: dict[int, list[tuple[int, float]]] = {}
        self.entry_id = -1
        self.columns = []

        self._load(json_path)
        self._build_columns(col_threshold)
        self._add_road_waypoints()
        self._add_entry_node()

    # ── load spots ────────────────────────────
    def _load(self, path):
        with open(path) as f:
            raw = json.load(f)
        for idx, item in enumerate(raw):
            pts = item["points"]
            cx = sum(p[0] for p in pts) / 4
            cy = sum(p[1] for p in pts) / 4
            self.spots[idx] = SpotNode(id=idx, cx=cx, cy=cy, points=pts)
            self.adjacency[idx] = []

    # ── group spots into columns ───────────────
    def _build_columns(self, col_threshold):
        ids = sorted(self.spots.keys(), key=lambda i: self.spots[i].cx)
        columns, current = [], [ids[0]]
        for sid in ids[1:]:
            if abs(self.spots[sid].cx - self.spots[current[-1]].cx) <= col_threshold:
                current.append(sid)
            else:
                columns.append(current)
                current = [sid]
        columns.append(current)
        self.columns = columns

        # sort each column by Y, connect consecutive spots within column
        for col in columns:
            col.sort(key=lambda i: self.spots[i].cy)
            for a, b in zip(col, col[1:]):
                self._link(a, b)

    # ── add road waypoints in aisles ───────────
    def _add_road_waypoints(self):
        """
        Place a vertical spine of waypoints in the aisle between each pair
        of adjacent columns. Waypoints are spaced every ~60px vertically
        and cover the full Y span of the lot.
        Spots connect to the nearest waypoint in their aisle instead of
        directly to spots in the next column.
        """
        all_spots = list(self.spots.values())
        y_min = min(s.cy for s in all_spots) - 30
        y_max = max(s.cy for s in all_spots) + 30
        y_step = 60

        wp_id = 1000  # waypoint IDs start at 1000 to avoid clash with spot IDs

        self.aisle_waypoints = []  # list of lists, one per aisle

        for c_idx in range(len(self.columns) - 1):
            col_a = self.columns[c_idx]
            col_b = self.columns[c_idx + 1]

            # aisle X = midpoint between the two column X centers
            xa = sum(self.spots[i].cx for i in col_a) / len(col_a)
            xb = sum(self.spots[i].cx for i in col_b) / len(col_b)
            aisle_x = (xa + xb) / 2

            # create waypoints along the aisle
            aisle_wps = []
            y = y_min
            while y <= y_max + y_step:
                self.spots[wp_id] = SpotNode(
                    id=wp_id, cx=aisle_x, cy=y,
                    points=[], status="road", is_waypoint=True
                )
                self.adjacency[wp_id] = []
                aisle_wps.append(wp_id)
                wp_id += 1
                y += y_step

            # connect waypoints vertically (the road spine)
            for a, b in zip(aisle_wps, aisle_wps[1:]):
                self._link(a, b)

            self.aisle_waypoints.append(aisle_wps)

            # connect each spot in col_a to its nearest aisle waypoint
            for sid in col_a:
                nearest = min(aisle_wps, key=lambda w: abs(self.spots[w].cy - self.spots[sid].cy))
                self._link(sid, nearest)

            # connect each spot in col_b to its nearest aisle waypoint
            for sid in col_b:
                nearest = min(aisle_wps, key=lambda w: abs(self.spots[w].cy - self.spots[sid].cy))
                self._link(sid, nearest)

        # also connect adjacent aisles horizontally at top and bottom
        for aw_a, aw_b in zip(self.aisle_waypoints, self.aisle_waypoints[1:]):
            self._link(aw_a[0],  aw_b[0])   # top
            self._link(aw_a[-1], aw_b[-1])  # bottom

    # ── entry node ────────────────────────────
    def _add_entry_node(self):
        eid = -1
        all_spots = [s for s in self.spots.values() if not s.is_waypoint]
        ex = min(s.cx for s in all_spots) - 60
        ey = min(s.cy for s in all_spots)
        self.spots[eid] = SpotNode(id=eid, cx=ex, cy=ey, points=[], is_waypoint=True)
        self.adjacency[eid] = []

        # entry connects to the first waypoint of the first aisle
        # and to the top spot of the first column
        if self.aisle_waypoints:
            self._link(eid, self.aisle_waypoints[0][0])
        if self.columns:
            self._link(eid, self.columns[0][0])

        self.entry_id = eid

    def _link(self, a, b):
        d = self._dist(a, b)
        if not any(nb == b for nb, _ in self.adjacency.get(a, [])):
            self.adjacency.setdefault(a, []).append((b, d))
            self.adjacency.setdefault(b, []).append((a, d))

    def _dist(self, a, b):
        sa, sb = self.spots[a], self.spots[b]
        return math.hypot(sa.cx - sb.cx, sa.cy - sb.cy)

    # ── occupancy ─────────────────────────────
    def update_occupancy(self, occupied_ids: set[int]):
        for sid, spot in self.spots.items():
            if spot.is_waypoint:
                continue
            spot.status = "occupied" if sid in occupied_ids else "free"

    def free_spots(self):
        return [s for s in self.spots.values()
                if not s.is_waypoint and s.status == "free"]

    def occupied_spots(self):
        return [s for s in self.spots.values()
                if not s.is_waypoint and s.status == "occupied"]

    # ── A* ────────────────────────────────────
    def astar(self, start_id, goal_id) -> Optional[list[int]]:
        open_set = [_PQItem(0.0, start_id)]
        came_from: dict[int, int] = {}
        g: dict[int, float] = {start_id: 0.0}

        def h(nid): return self._dist(nid, goal_id)

        while open_set:
            current = heapq.heappop(open_set).node_id
            if current == goal_id:
                return self._reconstruct(came_from, current)
            for nb, w in self.adjacency.get(current, []):
                tg = g[current] + w
                if tg < g.get(nb, math.inf):
                    came_from[nb] = current
                    g[nb] = tg
                    heapq.heappush(open_set, _PQItem(tg + h(nb), nb))
        return None

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))

    def best_route(self) -> Optional[dict]:
        free = self.free_spots()
        if not free:
            return None
        best_path, best_dist, best_spot = None, math.inf, None
        for spot in free:
            path = self.astar(self.entry_id, spot.id)
            if not path:
                continue
            dist = sum(self._dist(path[i], path[i+1]) for i in range(len(path)-1))
            if dist < best_dist:
                best_dist, best_path, best_spot = dist, path, spot
        if not best_path:
            return None
        return {
            "spot_id":  best_spot.id,
            "distance": round(best_dist, 1),
            "waypoints": [{"x": self.spots[n].cx, "y": self.spots[n].cy}
                          for n in best_path],
        }

    def to_json(self) -> dict:
        return {
            "spots": [
                {"id": s.id, "cx": s.cx, "cy": s.cy,
                 "points": s.points, "status": s.status}
                for s in self.spots.values()
                if not s.is_waypoint
            ],
            "edges": [
                {"from": a, "to": b}
                for a, nbs in self.adjacency.items()
                for b, _ in nbs
                if not self.spots[a].is_waypoint
                and not self.spots[b].is_waypoint
                and a < b
            ],
            "free_count":     len(self.free_spots()),
            "occupied_count": len(self.occupied_spots()),
            "total":          len([s for s in self.spots.values() if not s.is_waypoint]),
        }
