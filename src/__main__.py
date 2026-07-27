"""Entry point for the Synthetic Data Generation Component simulation framework.

Usage:
    python -m src
"""

import os
import json
import asyncio
import logging

from dotenv import load_dotenv

from .llm_client import OpenAIClient
from .config import TOPIC_TITLE, TOPIC_SUMMARY, SIMULATION_HORIZON_MINUTES
from .actors import build_actor_roster
from .simulation import run_simulation

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the full simulation pipeline.

    Loads environment variables, initialises the LLM client, builds the
    actor roster from behavioral parameters, executes the loop-based
    simulation, and persists the resulting discussion history to JSON.
    """
    llm_client = OpenAIClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    actors = build_actor_roster()

    H = await run_simulation(
        actors=actors,
        llm_client=llm_client,
        topic_title=TOPIC_TITLE,
        topic_summary=TOPIC_SUMMARY,
        T=SIMULATION_HORIZON_MINUTES,
    )

    # ── Persist ─────────────────────────────────────────────────────────────
    os.makedirs("session", exist_ok=True)
    with open("session/simulation_output.json", "w", encoding="utf-8") as f:
        json.dump(H.all_events(), f, indent=2, ensure_ascii=False)
    logger.info(f"📄 Saved {len(H)} events → session/simulation_output.json")


if __name__ == "__main__":
    asyncio.run(main())
