"""Build a reproducible static route file from one archived VD snapshot."""
from __future__ import annotations

import argparse
import os

from scenario_config import ROOT, resolve_scenario
from vd_provider import _read_net_routes, load_mapping, rates_from_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="heping")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--end", type=int, default=3600)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    sc = resolve_scenario(args.scenario)
    rows = load_mapping(sc.vd_mapping)
    with open(args.snapshot, "rb") as f:
        raw = f.read()
    rates, source_time = rates_from_snapshot(rows, raw)
    if not any(rates):
        raise RuntimeError("snapshot contains none of the selected scenario VD IDs")
    routes = _read_net_routes(sc.net_file, {r["SumoEdgeID"] for r in rows})
    output = args.output or os.path.join(ROOT, f"real_traffic_{sc.name}.rou.xml")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '    <vType id="passenger" vClass="passenger"/>',
        f'    <!-- archived VD source: {source_time or os.path.basename(args.snapshot)} -->',
    ]
    route_ids = {}
    for index, (source, edges) in enumerate(sorted(routes.items())):
        rid = f"vd_static_route_{index}"
        route_ids[source] = rid
        lines.append(f'    <route id="{rid}" edges="{" ".join(edges)}"/>')
    flows = 0
    for i, (row, vph) in enumerate(zip(rows, rates)):
        if vph < 1.0 or row["SumoEdgeID"] not in route_ids:
            continue
        lines.append(
            f'    <flow id="vd_static_{i}" type="passenger" '
            f'route="{route_ids[row["SumoEdgeID"]]}" begin="0" end="{args.end}" '
            f'vehsPerHour="{vph:.2f}" departLane="{row.get("departLane") or "best"}" '
            f'departSpeed="max"/>')
        flows += 1
    lines.extend(['</routes>', ''])
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"built {output}: {flows} flows, total={sum(rates):.1f} veh/h")


if __name__ == "__main__":
    main()
