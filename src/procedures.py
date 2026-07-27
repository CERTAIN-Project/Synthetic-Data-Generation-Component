"""Simulation procedures for voting and content generation.

Implements Algorithm 2 (ACTION) and Algorithm 3 (POST) from the paper.
Each procedure is called by the event loop when the corresponding event
type is dispatched for a given actor.
"""

from __future__ import annotations

import gc
import uuid
import random
import logging

from .prompts.anthropic_user_prompts import (
    COMMENT_PROMPT_ANTHROPIC,
    REPLY_PROMPT_ANTHROPIC,
    ACTION_PROMPT_ANTHROPIC,
)

from .models import ActorConfig, ActorState, SharedHistory
from .config import MODEL_NAME, MAX_TOKENS, M_CANDIDATES
from .formatting import (
    format_history_for_prompt,
    format_action_history,
    clean_agent_output,
    parse_candidate_selections,
    extract_reply_target,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Algorithm 2: ACTION(a_i)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def action_procedure(
    actor_idx: int,
    t: float,
    actors: list[ActorConfig],
    states: list[ActorState],
    H: SharedHistory,
    llm_client,
) -> None:
    """Execute the ACTION procedure (Algorithm 2) for a single actor.

    The agent reviews eligible posts it has not yet voted on, selects up
    to M candidates via the LLM, then applies Bernoulli gating to decide
    which votes are actually cast.

    Steps:
        1. Filter posts to those not yet voted on by this actor.
        2. Ask the LLM to select up to M candidates with vote types.
        3. For each candidate, draw ``u ~ U(0,1)``; execute the vote
           only if ``u > θ_i^action``.
        4. Record executed votes in both ``H`` and ``H_i^action``.

    Args:
        actor_idx: Index of the acting agent in the actors list.
        t: Current simulation time (minutes).
        actors: List of all actor configurations.
        states: List of all per-actor mutable states.
        H: The shared discussion history.
        llm_client: Async LLM client for generating selections.
    """
    actor = actors[actor_idx]
    state = states[actor_idx]

    all_posts = H.posts
    if not all_posts:
        logger.info(f"  ⏭  {actor.name}: No posts to vote on — skipping.")
        return

    eligible = [p for p in all_posts if p["id"] not in state.acted_on_ids]
    if not eligible:
        logger.info(f"  ⏭  {actor.name}: Already voted on everything — skipping.")
        return

    # ── Agent selects M candidates ──────────────────────────────────────────
    eligible_summary = "\n".join(
        f"[POST_ID={p['id']}] {p['actor']}: {p['text'][:150]}"
        for p in eligible
    )
    action_history_str = format_action_history(state.action_history, H)

    selection_prompt = ACTION_PROMPT_ANTHROPIC.format(
        eligible_posts=eligible_summary,
        action_history=action_history_str,
        m_candidates=min(M_CANDIDATES, len(eligible)),
    )

    response = await llm_client.achat(
        prompt=selection_prompt,
        system_prompt=actor.system_prompt,
        model_name=MODEL_NAME,
        temperature=0.4,
        max_tokens=1024,
    )

    candidates = parse_candidate_selections(response["text"], eligible)

    # ── Bernoulli gating per candidate ──────────────────────────────────────
    for candidate in candidates:
        u = random.random()
        if u > actor.theta_action:
            target = H.get_post(candidate["post_id"])
            if target is None:
                continue

            vote_type = candidate.get("vote_type", "upvote")

            H.add_vote(
                t=t,
                actor=actor.name,
                archetype=actor.archetype,
                age=random.randint(18, 70),  # Example demographic,
                nationality=random.choice(["US", "UK", "CA", "AU", "Other"]),  # Example demographic,  
                political_background=random.choice(["Liberal", "Conservative", "Moderate", "Other"]),  # Example demographic,
                fan_type=random.choice(["Die-hard", "Casual", "Fair-weather", "Other"]),  # Example demographic,
                target_id=target["id"],
                vote=vote_type,
            )
            state.action_history.append({"target_id": target["id"], "vote": vote_type})
            state.acted_on_ids.add(target["id"])

            logger.info(
                f"  🗳  {actor.name} → {vote_type} on [{target['actor']}] "
                f"(u={u:.2f} > θ={actor.theta_action})"
            )
        else:
            logger.debug(
                f"  ⊘  {actor.name}: vote suppressed (u={u:.2f} ≤ θ={actor.theta_action})"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Algorithm 3: POST(a_i)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def post_procedure(
    actor_idx: int,
    t: float,
    actors: list[ActorConfig],
    states: list[ActorState],
    H: SharedHistory,
    llm_client,
    topic_title: str,
    topic_summary: str,
) -> None:
    """Execute the POST procedure (Algorithm 3) for a single actor.

    The agent generates either a reply to an existing post or a new
    top-level comment, depending on a random draw against the actor's
    reply probability.

    Steps:
        1. Draw ``v ~ U(0,1)``. If ``v <= p_i^reply`` and posts exist,
           enter reply mode; otherwise enter new comment mode.
        2. In **reply mode**, present the 5 most recent posts by other
           actors as candidates and generate a reply via the LLM.
        3. In **new comment mode**, generate a standalone comment based
           on the topic, persona, and a randomly selected key belief.
        4. Optionally enrich the draft with web search (Expert only,
           50% probability).
        5. Record the post in both ``H`` and ``H_i^post``.

    Args:
        actor_idx: Index of the acting agent in the actors list.
        t: Current simulation time (minutes).
        actors: List of all actor configurations.
        states: List of all per-actor mutable states.
        H: The shared discussion history.
        llm_client: Async LLM client for content generation.
        topic_title: Title of the discussion topic.
        topic_summary: Summary description of the discussion topic.
    """
    actor = actors[actor_idx]
    state = states[actor_idx]

    discussion_history = format_history_for_prompt(H)

    v = random.random()
    is_reply = v <= actor.p_reply and len(H.posts) > 0
    reply_to_id: str | None = None

    if is_reply:
        # ── Reply mode ──────────────────────────────────────────────────────
        other_posts = [p for p in H.posts if p["actor"] != actor.name]
        if not other_posts:
            other_posts = H.posts

        candidates = other_posts[-5:]
        candidate_summary = "\n".join(
            f"[POST_ID={p['id']}] {p['actor']}: {p['text'][:200]}"
            for p in candidates
        )

        reply_prompt = REPLY_PROMPT_ANTHROPIC.format(
            discussion_title=topic_title,
            discussion_summary=topic_summary,
            candidate_posts=candidate_summary,
            discussion_history=discussion_history,
        )

        response = await llm_client.achat(
            prompt=reply_prompt,
            system_prompt=actor.system_prompt,
            model_name=MODEL_NAME,
            temperature=0.6,
            max_tokens=MAX_TOKENS,
        )
        comment_text = response["text"]
        reply_target = extract_reply_target(comment_text, candidates)
        reply_to_id = reply_target["id"] if reply_target else None
        comment_text = clean_agent_output(comment_text)

        logger.info(
            f"  💬 {actor.name} [REPLY → {reply_target['actor'] if reply_target else '?'}]: "
            f"{comment_text[:60]}..."
        )
    else:
        # ── New comment mode ────────────────────────────────────────────────
        comment_prompt = COMMENT_PROMPT_ANTHROPIC.format(
            discussion_title=topic_title,
            discussion_summary=topic_summary,
            discussion_history=discussion_history,
        )

        response = await llm_client.achat(
            prompt=comment_prompt,
            system_prompt=actor.system_prompt,
            model_name=MODEL_NAME,
            temperature=0.6,
            max_tokens=MAX_TOKENS,
        )
        comment_text = clean_agent_output(response["text"])
        logger.info(f"  💬 {actor.name} [NEW]: {comment_text[:60]}...")

    # ── Optional web search enrichment (Expert only) ────────────────────────
    if actor.has_web_search and random.random() < 0.5:
        comment_text = await _enrich_with_web_search(
            comment_text, topic_title, llm_client, actor
        )

    # ── Record in H and H_i^post ────────────────────────────────────────────
    post_id = str(uuid.uuid4())
    entry = H.add_post(
        post_id=post_id,
        t=t,
        actor=actor.name,
        archetype=actor.archetype,
        age=random.randint(18, 70),  # Example demographic,
        nationality=random.choice(["US", "UK", "CA", "AU", "Other"]),  # Example demographic,
        political_background=random.choice(["Liberal", "Conservative", "Moderate", "Other"]),  # Example demographic,
        fan_type=random.choice(["Die-hard", "Casual", "Fair-weather", "Other"]),  # Example demographic,
        text=comment_text,
        reply_to=reply_to_id,
    )
    state.post_history.append(entry)

    del response
    gc.collect()


# ── Expert-only web search enrichment ───────────────────────────────────────


async def _enrich_with_web_search(
    draft_text: str, query_seed: str, llm_client, actor: ActorConfig,
) -> str:
    """Invoke web search to retrieve evidence, then integrate into the draft.

    Asks the LLM to revise the draft by incorporating one or two concrete
    facts or statistics related to the query seed. Falls back to the
    original draft on failure.

    Args:
        draft_text: The original comment text to enrich.
        query_seed: A key belief or topic string to guide the search.
        llm_client: Async LLM client with web search tool support.
        actor: The actor configuration (used for system prompt).

    Returns:
        The enriched comment text, or the original draft if enrichment fails.
    """
    logger.info(f"  🔍 {actor.name}: web search for '{query_seed[:50]}'")
    try:
        prompt = (
            f"You previously drafted this comment:\n\n<draft>{draft_text}</draft>\n\n"
            f"Integrate one or two concrete facts or statistics from recent evidence "
            f"related to: '{query_seed}'. Keep your original tone and length. "
            f"Return ONLY the revised comment text."
        )
        response = await llm_client.achat(
            prompt=prompt,
            system_prompt=actor.system_prompt,
            model_name=MODEL_NAME,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            tools=[{"type": "web_search"}],
        )
        enriched = clean_agent_output(response["text"])
        if len(enriched) > 20:
            return enriched
    except Exception as e:
        logger.warning(f"  ⚠  Web search enrichment failed: {e}")
    return draft_text
