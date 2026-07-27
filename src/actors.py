"""Actor roster construction from behavioral parameters.

Builds the actor roster by combining the parameter table in config
with archetype-specific system prompts. Actor differentiation comes
from the event rates and gating thresholds, not individual profiles.
"""

from __future__ import annotations

import logging

from .models import ActorConfig
from .config import ACTOR_PARAMS, ARCHETYPE_TO_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_actor_roster() -> list[ActorConfig]:
    """Build the actor roster from the behavioral parameter table.

    Each entry in ``ACTOR_PARAMS`` is merged with the corresponding
    archetype system prompt to produce a fully configured ``ActorConfig``.

    Returns:
        List of ``ActorConfig`` instances ready for simulation.
    """
    roster: list[ActorConfig] = []
    for params in ACTOR_PARAMS:
        archetype = params["archetype"]
        roster.append(
            ActorConfig(
                name=params["name"],
                archetype=archetype,
                post_every_n_steps=params["post_every_n_steps"],
                action_every_n_steps=params["action_every_n_steps"],
                p_reply=params["p_reply"],
                theta_action=params["theta_action"],
                system_prompt=ARCHETYPE_TO_SYSTEM_PROMPT[archetype],
                has_web_search=params.get("has_web_search", False),
            )
        )
    logger.info(f"Built roster of {len(roster)} actors")
    return roster
