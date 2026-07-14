"""Scenario isolation for SUMO experiments.

The repository historically mixed Zhongxiao-Xinsheng and Heping-East assets
in the project root.  This module makes the selected road network explicit and
keeps RSU/VD artifacts tied to the same coordinate system.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Optional


ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Scenario:
    name: str
    sumocfg: str
    net_file: str
    rsu_positions: str
    vd_mapping: str
    vd_data_dir: str
    dynamic_routes: str


def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


SCENARIOS = {
    "heping": Scenario(
        name="heping",
        sumocfg=_p("heping.sumocfg"),
        net_file=_p("hepingeast2.net.xml"),
        rsu_positions=_p("scenarios", "heping", "rsu_positions.json"),
        vd_mapping=_p("scenarios", "heping", "vd_sumo_mapping.csv"),
        vd_data_dir=_p("traffic_data", "heping"),
        dynamic_routes=_p("scenarios", "heping", "vd_dynamic.rou.xml"),
    ),
    "zhongxiao": Scenario(
        name="zhongxiao",
        sumocfg=_p("osm.sumocfg"),
        net_file=_p("osm.net.xml.gz"),
        rsu_positions=_p("scenarios", "zhongxiao", "rsu_positions.json"),
        vd_mapping=_p("scenarios", "zhongxiao", "vd_sumo_mapping.csv"),
        vd_data_dir=_p("traffic_data", "zhongxiao"),
        dynamic_routes=_p("scenarios", "zhongxiao", "vd_dynamic.rou.xml"),
    ),
}


def cli_value(argv: Iterable[str], name: str, default: Optional[str] = None) -> Optional[str]:
    args = list(argv)
    prefix = name + "="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == name and index + 1 < len(args):
            return args[index + 1]
    return default


def resolve_scenario(name: Optional[str] = None, cfg: Optional[str] = None) -> Scenario:
    """Resolve a named scenario, optionally overriding only its SUMO cfg."""
    key = (name or "heping").lower()
    if key not in SCENARIOS:
        raise ValueError(f"unknown scenario {key!r}; choose {sorted(SCENARIOS)}")
    base = SCENARIOS[key]
    if not cfg:
        return base
    cfg_path = cfg if os.path.isabs(cfg) else _p(cfg)
    return Scenario(base.name, cfg_path, base.net_file, base.rsu_positions,
                    base.vd_mapping, base.vd_data_dir, base.dynamic_routes)


def scenario_cli(argv: Iterable[str], use_sumo: bool) -> tuple[Optional[Scenario], str]:
    """Return selected scenario and VD mode for experiment scripts."""
    if not use_sumo:
        return None, "off"
    name = cli_value(argv, "--scenario", "heping")
    cfg = cli_value(argv, "--cfg")
    mode = (cli_value(argv, "--vd-mode", "static") or "static").lower()
    if mode not in {"static", "replay", "live"}:
        raise ValueError("--vd-mode must be static, replay, or live")
    return resolve_scenario(name, cfg), mode


def vd_split_cli(argv: Iterable[str], use_sumo: bool, vd_mode: str,
                 default: str = "all") -> str:
    """Parse the chronological replay partition selected by an experiment."""
    if not use_sumo or vd_mode != "replay":
        return "all"
    split = (cli_value(argv, "--vd-split", default) or default).lower()
    split = "validation" if split == "val" else split
    if split not in {"all", "train", "validation", "test"}:
        raise ValueError("--vd-split must be all, train, validation, or test")
    return split


def model_suffix(*, local_critic: bool = False, priority: bool = False,
                 twin_quality: bool = False, scenario: Optional[Scenario] = None,
                 vd_mode: str = "static") -> str:
    suffix = f"_{scenario.name}" if scenario is not None else ""
    if scenario is not None and vd_mode != "static":
        suffix += f"_{vd_mode}"
    suffix += "_localcritic" if local_critic else ""
    suffix += "_prio" if priority else ""
    suffix += "_tq" if twin_quality else ""
    return suffix


def env_scenario_kwargs(scenario: Optional[Scenario], vd_mode: str = "off",
                        vd_split: str = "all") -> dict:
    if scenario is None:
        return {}
    return {
        "scenario": scenario.name,
        "cfg": scenario.sumocfg,
        "rsu_path": scenario.rsu_positions,
        "vd_mode": vd_mode,
        "vd_split": vd_split,
        "vd_mapping": scenario.vd_mapping,
        "vd_data_dir": scenario.vd_data_dir,
        "vd_net_file": scenario.net_file,
        "vd_dynamic_routes": scenario.dynamic_routes,
    }
