"""Prompt formatting, history rendering, and agent output parsing.

Converts simulation state into prompt-friendly strings and parses
raw LLM output back into structured data for the simulation loop.
"""

from __future__ import annotations

import re
import json
import random
import logging
from typing import Optional

from .models import SharedHistory
from .config import M_CANDIDATES

logger = logging.getLogger(__name__)


# ── History → prompt string ─────────────────────────────────────────────────


def format_history_for_prompt(H: SharedHistory, max_posts: int = 30) -> str:
    """Render the most recent posts from the shared history as a prompt string.

    Each post is formatted as ``[actor] (replying to parent): text``. If
    there are no posts yet, returns a placeholder XML tag.

    Args:
        H: The shared discussion history.
        max_posts: Maximum number of recent posts to include.

    Returns:
        Newline-separated string of formatted posts.
    """
    posts = H.posts
    if not posts:
        return "<no_posts_yet />"
    recent = posts[-max_posts:]
    lines = []
    for p in recent:
        prefix = f"[{p['actor']}]"
        if p.get("reply_to"):
            parent = H.get_post(p["reply_to"])
            if parent:
                prefix += f" (replying to {parent['actor']})"
        lines.append(f"{prefix}: {p['text']}")
    return "\n".join(lines)


def format_action_history(action_history: list[dict], H: SharedHistory) -> str:
    """Render an actor's voting history as a prompt string.

    Each entry shows the vote type, the original post author, and a
    truncated snippet of the post text.

    Args:
        action_history: List of vote records from ``ActorState.action_history``.
        H: The shared discussion history (used to look up post details).

    Returns:
        Newline-separated string of past votes, or a placeholder tag if empty.
    """
    if not action_history:
        return "<no_actions_yet />"
    lines = []
    for a in action_history:
        target = H.get_post(a["target_id"])
        snippet = (target["text"][:80] + "...") if target else "?"
        who = target["actor"] if target else "?"
        lines.append(f"- {a['vote']} on post by {who}: \"{snippet}\"")
    return "\n".join(lines)


# ── Agent output cleaning ──────────────────────────────────────────────────


def clean_agent_output(text: str) -> str:
    """Strip meta-tags, POST_ID references, and code fences from agent output.

    Args:
        text: Raw text returned by the LLM.

    Returns:
        Cleaned plain-text string.
    """
    text = re.sub(r"\[POST_ID=[^\]]*\]", "", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


# ── Candidate selection parsing ─────────────────────────────────────────────


def parse_candidate_selections(
    agent_response: str, eligible: list[dict]
) -> list[dict]:
    """Parse the LLM's JSON vote selections into structured candidates.

    Expects a JSON array of ``{"post_id": ..., "vote_type": ...}`` objects.
    Strips code fences and filters to valid eligible post IDs. Falls back
    to random sampling if parsing fails.

    Args:
        agent_response: Raw text from the LLM containing a JSON array.
        eligible: List of post dicts the actor is allowed to vote on.

    Returns:
        List of dicts with ``"post_id"`` and ``"vote_type"`` keys.
    """
    try:
        text = agent_response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip().removeprefix("json").strip()

        candidates = json.loads(text)
        if isinstance(candidates, list):
            eligible_ids = {p["id"] for p in eligible}
            return [c for c in candidates if c.get("post_id") in eligible_ids]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    logger.warning("  ⚠  Could not parse agent selection — random fallback.")
    k = min(M_CANDIDATES, len(eligible))
    return [
        {"post_id": p["id"], "vote_type": random.choice(["upvote", "downvote"])}
        for p in random.sample(eligible, k)
    ]


# ── Reply target extraction ────────────────────────────────────────────────


def extract_reply_target(
    agent_output: str, candidates: list[dict]
) -> Optional[dict]:
    """Identify which candidate post the agent chose to reply to.

    Searches the agent's output for a matching POST_ID or actor name.
    Falls back to the most recent candidate if no match is found.

    Args:
        agent_output: Raw text from the LLM reply.
        candidates: List of candidate post dicts that were offered.

    Returns:
        The matched candidate dict, or the last candidate as a fallback.
        Returns ``None`` if candidates is empty.
    """
    for c in candidates:
        if c["id"] in agent_output:
            return c
    for c in candidates:
        if c["actor"].lower() in agent_output.lower():
            return c
    return candidates[-1] if candidates else None
