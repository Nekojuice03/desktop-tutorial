"""Archive Taipei section-level VD snapshots for reproducible replay."""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import os
import time

from scenario_config import resolve_scenario
from vd_provider import LIVE_SECTION_URL, load_mapping, rates_from_snapshot


def fetch(url: str) -> bytes:
    from urllib.request import Request, urlopen
    request = Request(url, headers={"User-Agent": "NTUT-DigitalTwin/2.0"})
    with urlopen(request, timeout=20) as response:
        raw = response.read()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="heping")
    p.add_argument("--interval", type=int, default=300, help="wall-clock seconds")
    p.add_argument("--count", type=int, default=1, help="0 means continuous")
    p.add_argument("--url", default=LIVE_SECTION_URL)
    args = p.parse_args()
    scenario = resolve_scenario(args.scenario)
    rows = load_mapping(scenario.vd_mapping)
    os.makedirs(scenario.vd_data_dir, exist_ok=True)
    n = 0
    try:
        while args.count == 0 or n < args.count:
            try:
                raw = fetch(args.url)
                rates, source_time = rates_from_snapshot(rows, raw)
                if not any(rates):
                    raise RuntimeError(
                        "download succeeded but no mapped VD IDs were present")
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = os.path.join(scenario.vd_data_dir, f"VD_SECTION_{stamp}.xml")
                with open(out, "wb") as f:
                    f.write(raw)
                print(f"saved {out} | source={source_time or 'unknown'} | "
                      f"total={sum(rates):.1f} veh/h", flush=True)
                n += 1
            except Exception as exc:
                if args.count != 0:
                    raise
                print(f"warning: VD fetch failed ({exc}); retrying in "
                      f"{min(60, max(1, args.interval))}s", flush=True)
                time.sleep(min(60, max(1, args.interval)))
                continue
            if args.count == 0 or n < args.count:
                time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        print(f"\nstopped cleanly; collected {n} snapshot(s) in this run")


if __name__ == "__main__":
    main()
