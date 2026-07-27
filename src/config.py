"""Simulation configuration - constants, actor step settings, prompt mapping."""

from __future__ import annotations

import os

from .prompts.anthropic_system_prompts import CASUAL_USER_PROMPT_ANTHROPIC

# LLM

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_TOKENS = 4096

# Simulation

SIMULATION_HORIZON_MINUTES = 0.6  # T
SIMULATION_STEP_MINUTES = 0.1
M_CANDIDATES = 3  # max posts the agent selects for voting

# Topic

TOPIC_TITLE = "Giannakopoulos dissapointment with panathinaikos not qualifying for the Euroleague Final Four"
TOPIC_SUMMARY = (
    "A public deliberation on the disappointment of Giannakopoulos with Panathinaikos not qualifying for the Euroleague Final Four, exploring fan perspectives, "
    "team dynamics, and the cultural impact on the local community."
    "This topic invites discussion on the emotional and social implications of sports outcomes, as well as the broader cultural significance of basketball in the region."
    "Participants will share their views on the factors contributing to the team's performance, the reactions of fans and stakeholders, and the potential future of Panathinaikos in European basketball."
)

# Archetype -> system prompt

ARCHETYPE_TO_SYSTEM_PROMPT: dict[str, str] = {
    "citizen_b2c": CASUAL_USER_PROMPT_ANTHROPIC,
}

# Actor behavioral parameters (Table 1)

ACTOR_PARAMS: list[dict] = [
    {
        "name": "Casual User 1",
        "archetype": "citizen_b2c",
        "post_every_n_steps": 2,
        "action_every_n_steps": 3,
        "p_reply": 0.45,
        "theta_action": 0.35,
    },
    {
        "name": "Casual User 2",
        "archetype": "citizen_b2c",
        "post_every_n_steps": 3,
        "action_every_n_steps": 4,
        "p_reply": 0.40,
        "theta_action": 0.40,
    },
    {
        "name": "Casual User 3",
        "archetype": "citizen_b2c",
        "post_every_n_steps": 2,
        "action_every_n_steps": 3,
        "p_reply": 0.50,
        "theta_action": 0.30,
    },
    {
        "name": "Casual User 4",
        "archetype": "citizen_b2c",
        "post_every_n_steps": 4,
        "action_every_n_steps": 5,
        "p_reply": 0.35,
        "theta_action": 0.45,
    },
]
