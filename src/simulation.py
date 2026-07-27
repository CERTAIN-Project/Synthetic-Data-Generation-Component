"""Algorithm 1: Agentic Simulation Cycle — simple loop-based scheduler.

Implements the main simulator using fixed time steps and per-actor
interval checks over a configurable time horizon.
"""

from __future__ import annotations

import logging

from .models import (
    ActorConfig,
    ActorState,
    SharedHistory,
)
from .config import SIMULATION_HORIZON_MINUTES, SIMULATION_STEP_MINUTES
from .procedures import action_procedure, post_procedure

logger = logging.getLogger(__name__)


async def run_simulation(
    actors: list[ActorConfig],
    llm_client,
    topic_title: str,
    topic_summary: str,
    T: float = SIMULATION_HORIZON_MINUTES,
) -> SharedHistory:
    """Run the Agentic Simulation Cycle (Algorithm 1).

    The simulation advances in fixed time steps. On each step, every actor
    is checked against a simple post interval and action interval. When an
    actor reaches its interval, the corresponding procedure is executed.

    Args:
        actors: List of configured actor instances.
        llm_client: Async LLM client for content generation and voting.
        topic_title: Title of the discussion topic.
        topic_summary: Summary description of the discussion topic.
        T: Simulation time horizon in minutes. Events beyond this
            time are discarded.

    Returns:
        The shared discussion history containing all posts and votes
        in chronological order.
    """
    actor_count = len(actors)
    states: list[ActorState] = [ActorState() for _ in range(actor_count)]
    H = SharedHistory()

    total_steps = max(1, int(T / SIMULATION_STEP_MINUTES))

    logger.info(
        f"🚀 Simulation started — {actor_count} actors, T={T} min, "
        f"step={SIMULATION_STEP_MINUTES} min, total_steps={total_steps}"
    )

    total_posts = 0
    total_actions = 0

    # ── Main simulation loop ────────────────────────────────────────────────
    for step_index in range(1, total_steps + 1):
        current_time = min(step_index * SIMULATION_STEP_MINUTES, T)

        for actor_idx, actor in enumerate(actors):
            if actor.post_every_n_steps > 0 and step_index % actor.post_every_n_steps == 0:
                logger.info(f"[t={current_time:6.2f}] 📝 POST — {actor.name}")
                await post_procedure(
                    actor_idx=actor_idx,
                    t=current_time,
                    actors=actors,
                    states=states,
                    H=H,
                    llm_client=llm_client,
                    topic_title=topic_title,
                    topic_summary=topic_summary,
                )
                total_posts += 1

            if actor.action_every_n_steps > 0 and step_index % actor.action_every_n_steps == 0:
                logger.info(f"[t={current_time:6.2f}] 🗳  ACTION — {actor.name}")
                await action_procedure(
                    actor_idx=actor_idx,
                    t=current_time,
                    actors=actors,
                    states=states,
                    H=H,
                    llm_client=llm_client,
                )
                total_actions += 1

    logger.info(f"✅ Done — {total_posts} posts, {total_actions} action events, {len(H)} total in H")
    return H
