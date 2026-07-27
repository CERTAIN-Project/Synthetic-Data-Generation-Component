"""
Synthetic Data Generation Component Orchestration Package
=============================
Agentic simulation cycle for generating realistic deliberation data.

    python -m orchestration

Modules:
    models       — ActorConfig, ActorState, SharedHistory
    config       — constants, actor parameter table, prompt mapping
    actors       — persona loading, roster construction
    formatting   — prompt rendering, output parsing
    procedures   — Algorithm 2 (ACTION) and Algorithm 3 (POST)
    simulation   — Algorithm 1 (simple main loop)
    result_exporter — SimulationResultExporter, MinIOResultUploader, AnonymousActorSimulationHandler
"""

from .models import ActorConfig, ActorState, SharedHistory
from .actors import build_actor_roster
from .simulation import run_simulation
from .result_exporter import SimulationResultExporter, MinIOResultUploader, AnonymousActorSimulationHandler
from .name_generation import generate_participant_metadata

__all__ = [
    "ActorConfig",
    "ActorState",
    "SharedHistory",
    "build_actor_roster",
    "run_simulation",
    "SimulationResultExporter",
    "MinIOResultUploader",
    "AnonymousActorSimulationHandler",
    "generate_participant_metadata",
]

