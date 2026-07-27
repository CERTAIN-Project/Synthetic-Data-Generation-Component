"""Core data structures for the Synthetic Data Generation Component simulation.

Defines the immutable configuration objects, mutable per-actor state,
and the shared discussion history used by the simple loop-based scheduler.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class ActorConfig:
    """Behavioral parameters for a single actor (Table 1 in the paper).

    Attributes:
        name: Display name of the actor.
        archetype: ``citizen_b2c`` .
        post_every_n_steps: Run the post procedure every N loop steps.
        action_every_n_steps: Run the action procedure every N loop steps.
        p_reply: Probability of replying to an existing post vs. creating new.
        theta_action: Bernoulli gating threshold for voting selectivity.
        system_prompt: Archetype-specific system prompt for the LLM.
        has_web_search: Whether this actor can enrich posts via web search.
    """

    name: str
    archetype: str
    post_every_n_steps: int
    action_every_n_steps: int
    p_reply: float              # p_i^reply — probability of replying vs. new comment
    theta_action: float         # θ_i^action — action selectivity threshold
    system_prompt: str = ""
    has_web_search: bool = False


@dataclass
class ActorState:
    """Per-actor mutable state tracking posts and votes.

    Attributes:
        post_history: Posts created by this actor (H_i^post).
        action_history: Votes cast by this actor (H_i^action).
        acted_on_ids: Set of post IDs already voted on (deduplication).
    """

    post_history: list[dict] = field(default_factory=list)
    action_history: list[dict] = field(default_factory=list)
    acted_on_ids: set = field(default_factory=set)


class SharedHistory:
    """Append-only, chronologically ordered shared discussion history (H).

    Every entry is a dict with at minimum:
    ``{"id", "t", "type", "actor", "archetype"}``.

    Posts additionally carry: ``"text"``, ``"reply_to"`` (id or ``None``).
    Votes additionally carry: ``"target_id"``, ``"vote"``.

    Attributes:
        _events: Flat chronological list of all events.
        _posts_by_id: Index for O(1) post lookup by UUID.
    """

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._posts_by_id: dict[str, dict] = {}

    def add_post(
        self,
        *,
        post_id: str,
        t: float,
        actor: str,
        archetype: str,
        text: str,
        age: int,
        nationality: str,
        political_background: str,
        fan_type: str,
        reply_to: str | None = None,
    ) -> dict:
        """Record a new post in the shared history.

        Args:
            post_id: Unique identifier for the post.
            t: Simulation time when the post was created.
            actor: Name of the actor who authored the post.
            archetype: Archetype of the authoring actor.
            text: The generated comment text.
            reply_to: ID of the parent post if this is a reply,
                or ``None`` for top-level comments.

        Returns:
            The event dict that was appended to the history.
        """
        entry = {
            "id": post_id,
            "t": round(t, 3),
            "type": "post",
            "actor": actor,
            "archetype": archetype,
            "text": text,
            "age": age,
            "nationality": nationality,
            "political_background": political_background,
            "fan_type": fan_type,
            "reply_to": reply_to,
        }
        self._events.append(entry)
        self._posts_by_id[post_id] = entry
        return entry

    def add_vote(
        self,
        *,
        t: float,
        actor: str,
        archetype: str,
        age: int,
        nationality: str,
        political_background: str,
        fan_type: str,
        target_id: str,
        vote: str,
    ) -> dict:
        """Record a vote in the shared history.

        Args:
            t: Simulation time when the vote was cast.
            actor: Name of the voting actor.
            archetype: Archetype of the voting actor.
            target_id: ID of the post being voted on.
            vote: Vote type, typically ``"upvote"`` or ``"downvote"``.

        Returns:
            The event dict that was appended to the history.
        """
        entry = {
            "id": str(uuid.uuid4()),
            "t": round(t, 3),
            "type": "vote",
            "actor": actor,
            "archetype": archetype,
            "age": age,
            "nationality": nationality,
            "political_background": political_background,
            "fan_type": fan_type,
            "target_id": target_id,
            "vote": vote,
        }
        self._events.append(entry)
        return entry

    @property
    def posts(self) -> list[dict]:
        """Return all post events from the history."""
        return [e for e in self._events if e["type"] == "post"]

    @property
    def votes(self) -> list[dict]:
        """Return all vote events from the history."""
        return [e for e in self._events if e["type"] == "vote"]

    def get_post(self, post_id: str) -> dict | None:
        """Look up a post by its UUID.

        Args:
            post_id: The unique identifier of the post.

        Returns:
            The post event dict, or ``None`` if not found.
        """
        return self._posts_by_id.get(post_id)

    def all_events(self) -> list[dict]:
        """Return a shallow copy of all events in chronological order."""
        return list(self._events)

    def __len__(self) -> int:
        """Return the total number of events (posts + votes)."""
        return len(self._events)
