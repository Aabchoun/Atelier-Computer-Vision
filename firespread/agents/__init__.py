"""System agents and their registry."""

from __future__ import annotations

from typing import Any

from firespread.agents.burned_area import BurnedAreaAgent
from firespread.agents.environment import EnvironmentAgent
from firespread.agents.fire import FireAgent
from firespread.agents.simulation import SimulationAgent
from firespread.agents.smoke import SmokeAgent
from firespread.agents.spread import SpreadAgent
from firespread.agents.vegetation import VegetationAgent
from firespread.common.agent import Agent

AGENT_CLASSES: dict[str, type[Agent]] = {
    cls.name: cls
    for cls in (FireAgent, SmokeAgent, VegetationAgent, BurnedAreaAgent, EnvironmentAgent, SpreadAgent, SimulationAgent)
}

# Execution order used by the coordinator. Fire runs first: its fused layer is used by smoke.
PERCEPTION_AGENTS = ("fire", "smoke", "vegetation", "burned_area", "environment")
FORECAST_AGENTS = ("spread", "simulation")


def build_agents(config: dict[str, Any]) -> dict[str, Agent]:
    """Instantiate the enabled agents with their config section (+ the shared fire threshold)."""
    agents_cfg = config.get("agents", {})
    enabled = agents_cfg.get("enabled", list(AGENT_CLASSES))
    unknown = set(enabled) - set(AGENT_CLASSES)
    if unknown:
        raise ValueError(f"unknown agents in config: {sorted(unknown)}; available: {sorted(AGENT_CLASSES)}")
    shared = {"fire_threshold": config.get("coordinator", {}).get("fire_threshold", 0.5)}
    return {name: AGENT_CLASSES[name]({**shared, **(agents_cfg.get(name) or {})}) for name in enabled}


__all__ = ["AGENT_CLASSES", "FORECAST_AGENTS", "PERCEPTION_AGENTS", "build_agents", *[c.__name__ for c in AGENT_CLASSES.values()]]
