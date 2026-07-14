"""Live/replay Taipei VD traffic provider for a running TraCI connection.

The provider owns no SUMO process.  It injects vehicles through the same TraCI
connection used by the RL environment, which prevents the former two-process
calibrator prototype from racing with ``TraciWorld``.
"""
from __future__ import annotations

from collections import defaultdict, deque
import csv
from datetime import datetime, timezone
import gzip
import os
import re
import time
import xml.etree.ElementTree as ET


LIVE_SECTION_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
REPLAY_SPLITS = ("all", "train", "validation", "test")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(elem, names, default=""):
    wanted = {n.lower() for n in names}
    for node in elem.iter():
        if _local(node.tag).lower() in wanted and node.text:
            return node.text.strip()
    return default


def parse_vd_xml(raw: bytes | str) -> tuple[dict, dict, str | None]:
    """Parse section-level TotalVol and device/lane small-vehicle volume."""
    if isinstance(raw, bytes):
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = raw
    root = ET.fromstring(text)
    sections, lanes = {}, {}
    source_time = _child_text(root, ("UpdateTime", "DataCollectTime", "Time"), "") or None
    for elem in root.iter():
        kind = _local(elem.tag)
        if kind == "SectionData":
            sid = _child_text(elem, ("SectionId", "SectionID"))
            vol = _child_text(elem, ("TotalVol", "Volume"))
            interval = _child_text(elem, ("DataCollectTimeInterval", "TimeInterval"), "5")
            if sid and vol:
                try:
                    sections[sid] = (float(vol), max(float(interval), 1.0))
                except ValueError:
                    pass
        elif kind == "VDDevice":
            did = _child_text(elem, ("DeviceID", "VDID"))
            interval = _child_text(elem, ("DataCollectTimeInterval", "TimeInterval"), "5")
            if not did:
                continue
            for lane in elem.iter():
                if _local(lane.tag) != "LaneData":
                    continue
                lane_no = _child_text(lane, ("LaneNO", "LaneNo"), "0")
                small = _child_text(lane, ("Svolume", "Volume"), "0")
                try:
                    lanes[(did, int(float(lane_no)))] = (
                        float(small), max(float(interval), 1.0))
                except ValueError:
                    pass
    return sections, lanes, source_time


def load_mapping(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    section_groups = defaultdict(list)
    lane_groups = defaultdict(list)
    for row in rows:
        did = row["DeviceID"].strip()
        lane_no = int(float(row.get("LaneNO") or 0))
        section_groups[did].append(row)
        lane_groups[(did, lane_no)].append(row)

    def assign(groups, field):
        for group in groups.values():
            explicit = [float(r.get("FlowShare") or 0.0) for r in group]
            if any(explicit):
                total = sum(explicit)
                for row, value in zip(group, explicit):
                    row[field] = value / total if total else 0.0
            else:
                for row in group:
                    row[field] = 1.0 / len(group)

    # Section TotalVol is one aggregate per detector/section, whereas device
    # data has a distinct measurement for each lane.  Keeping two shares
    # prevents a six-lane device from being divided by six a second time.
    assign(section_groups, "_section_share")
    assign(lane_groups, "_lane_share")
    return rows


def rates_from_snapshot(rows: list[dict], raw: bytes | str,
                        passenger_ratio: float = 0.75) -> tuple[list[float], str | None]:
    sections, lanes, source_time = parse_vd_xml(raw)
    rates = []
    for row in rows:
        did = row["DeviceID"].strip()
        lane_no = int(float(row.get("LaneNO") or 0))
        if (did, lane_no) in lanes:
            volume, minutes = lanes[(did, lane_no)]
            vph = volume * 60.0 / minutes * row["_lane_share"]
        elif did in sections:
            volume, minutes = sections[did]
            vph = volume * 60.0 / minutes * passenger_ratio * row["_section_share"]
        else:
            vph = 0.0
        rates.append(max(0.0, vph))
    return rates, source_time


def _read_net_routes(path: str, sources: set[str]) -> dict[str, list[str]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        root = ET.parse(f).getroot()
    edges = {e.get("id") for e in root.findall("edge")
             if e.get("id") and not e.get("function")}
    graph = defaultdict(list)
    for conn in root.findall("connection"):
        a, b = conn.get("from"), conn.get("to")
        if a in edges and b in edges and b not in graph[a]:
            graph[a].append(b)
    exits = {e for e in edges if not graph[e]}
    result = {}
    for source in sources:
        if source not in edges:
            continue
        if source in exits:
            # A detector may be mapped to an outbound boundary segment.  A
            # one-edge SUMO route is valid and still models measured boundary
            # traffic traversing that final segment.
            result[source] = [source]
            continue
        q = deque([(source, [source])])
        seen = {source}
        candidates = []
        while q:
            edge, route = q.popleft()
            if edge in exits and len(route) > 1:
                candidates.append(route)
            for nxt in graph[edge]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, route + [nxt]))
        if candidates:
            result[source] = max(candidates, key=len)
    return result


def select_replay_files(files: list[str], split: str) -> list[str]:
    """Chronologically partition archived snapshots without data leakage."""
    split = "validation" if split == "val" else split
    if split not in REPLAY_SPLITS:
        raise ValueError(f"replay split must be one of {REPLAY_SPLITS}")
    files = sorted(files)
    if split == "all":
        return files
    if len(files) < 3:
        raise ValueError(
            f"VD split {split!r} needs at least 3 snapshots; found {len(files)}")
    n = len(files)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    bounds = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    lo, hi = bounds[split]
    return files[lo:hi]


class VDTrafficProvider:
    def __init__(self, mode: str, mapping_path: str, data_dir: str,
                 net_file: str, update_interval: float = 300.0,
                 live_url: str = LIVE_SECTION_URL,
                 replay_split: str = "all", replay_episode: int = 0):
        if mode not in {"replay", "live"}:
            raise ValueError("VDTrafficProvider mode must be replay or live")
        self.mode = mode
        self.mapping_path = mapping_path
        self.data_dir = data_dir
        self.net_file = net_file
        self.update_interval = float(update_interval)
        self.live_url = live_url
        self.replay_split = "validation" if replay_split == "val" else replay_split
        self.replay_episode = int(replay_episode)
        self.rows = load_mapping(mapping_path)
        self.routes = _read_net_routes(net_file,
                                       {r["SumoEdgeID"] for r in self.rows})
        missing = sorted({r["SumoEdgeID"] for r in self.rows} - set(self.routes))
        if missing:
            raise ValueError(f"VD mapping has unroutable edges for selected net: {missing}")
        self.rates = [0.0] * len(self.rows)
        self.accum = [1.0] * len(self.rows)  # seed each mapped boundary at episode start
        self.route_ids = {}
        self.replay_files = []
        self.replay_index = -1
        self.next_sim_update = 0.0
        self.next_wall_update = 0.0
        self.last_sim_update = 0.0
        self.last_wall_update = 0.0
        self.source_time = None
        self.data_age_s = 0.0
        self.update_count = 0
        self.fetch_failures = 0
        self.injected = 0
        self.inject_failures = 0
        self._serial = 0

    @property
    def dynamic_routes(self) -> bool:
        return True

    def attach(self, traci):
        for source, edges in self.routes.items():
            rid = "vd_route_" + re.sub(r"[^A-Za-z0-9_]", "_", source)
            self.route_ids[source] = rid
            try:
                traci.route.add(rid, edges)
            except Exception:
                # A reconnect may retain route definitions in some SUMO builds.
                pass
        if self.mode == "replay":
            os.makedirs(self.data_dir, exist_ok=True)
            all_files = sorted(
                os.path.join(self.data_dir, name) for name in os.listdir(self.data_dir)
                if name.lower().endswith(".xml")
            )
            self.replay_files = select_replay_files(all_files, self.replay_split)
            if not self.replay_files:
                raise FileNotFoundError(
                    f"no replay VD snapshots in {self.data_dir}; run collect_vd.py first")
            # One 300-second episode represents one five-minute VD aggregate.
            # Rotate the starting snapshot across resets so training uses the
            # whole selected partition instead of always replaying file zero.
            self.replay_index = self.replay_episode % len(self.replay_files) - 1
            self._advance_replay()
            self.next_sim_update = self.update_interval
        else:
            self._fetch_live()
            self.last_wall_update = time.monotonic()
            self.next_wall_update = self.last_wall_update + self.update_interval

    def _advance_replay(self):
        for _ in range(max(1, len(self.replay_files))):
            self.replay_index = (self.replay_index + 1) % len(self.replay_files)
            path = self.replay_files[self.replay_index]
            with open(path, "rb") as f:
                raw = f.read()
            rates, source_time = rates_from_snapshot(self.rows, raw)
            if any(rates):
                self.rates, self.source_time = rates, source_time or os.path.basename(path)
                self.update_count += 1
                self.data_age_s = 0.0
                return
        raise ValueError("replay snapshots contain none of the selected scenario VD IDs")

    def _fetch_live(self):
        try:
            from urllib.request import Request, urlopen
            request = Request(self.live_url,
                              headers={"User-Agent": "NTUT-DigitalTwin/2.0"})
            with urlopen(request, timeout=20) as response:
                raw = response.read()
            rates, source_time = rates_from_snapshot(self.rows, raw)
            if not any(rates):
                raise ValueError("live feed has no matching mapped VD values")
            self.rates, self.source_time = rates, source_time
            self.update_count += 1
            self.data_age_s = 0.0
            os.makedirs(self.data_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(self.data_dir, f"VD_SECTION_{stamp}.xml"), "wb") as f:
                f.write(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        except Exception:
            self.fetch_failures += 1
            if not any(self.rates):
                raise

    def before_step(self, traci, sim_time: float, dt: float):
        if self.mode == "replay" and sim_time >= self.next_sim_update:
            self._advance_replay()
            self.last_sim_update = sim_time
            self.next_sim_update = sim_time + self.update_interval
        elif self.mode == "live" and time.monotonic() >= self.next_wall_update:
            self._fetch_live()
            self.last_wall_update = time.monotonic()
            self.next_wall_update = self.last_wall_update + self.update_interval

        # This is ingestion age (time since the provider accepted a snapshot).
        # The public section feed currently does not expose a reliable source
        # timestamp, so it must not be presented as end-to-end sensor AoI.
        if self.mode == "replay":
            self.data_age_s = max(0.0, sim_time - self.last_sim_update)
        else:
            self.data_age_s = max(0.0, time.monotonic() - self.last_wall_update)

        for i, (row, vph) in enumerate(zip(self.rows, self.rates)):
            self.accum[i] += vph * dt / 3600.0
            while self.accum[i] >= 1.0:
                self.accum[i] -= 1.0
                source = row["SumoEdgeID"]
                vid = f"vd_{i}_{self._serial}"
                self._serial += 1
                try:
                    traci.vehicle.add(
                        vid, self.route_ids[source], typeID="passenger",
                        departLane=str(row.get("departLane") or "best"),
                        departSpeed="max",
                    )
                    self.injected += 1
                except Exception:
                    self.inject_failures += 1

    def status(self) -> dict:
        return {
            "vd_mode": self.mode,
            "vd_split": self.replay_split if self.mode == "replay" else "live",
            "vd_source_time": self.source_time,
            "vd_data_age_s": float(self.data_age_s),
            "vd_updates": self.update_count,
            "vd_fetch_failures": self.fetch_failures,
            "vd_injected": self.injected,
            "vd_inject_failures": self.inject_failures,
            "vd_total_vph": float(sum(self.rates)),
            "vd_snapshot_count": len(self.replay_files),
            "vd_snapshot_index": self.replay_index if self.mode == "replay" else None,
        }
