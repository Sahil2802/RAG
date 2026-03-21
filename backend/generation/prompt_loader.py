import yaml
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

# Resolve path relative to this file: backend/config/prompts.yaml
PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    """Load all prompts from the YAML file. Cached per process."""
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Prompts file not found: {PROMPTS_PATH}")

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    if not prompts or "active_prompt" not in prompts:
        raise ValueError("prompts.yaml must contain an 'active_prompt' key")

    logger.info(f"Loaded prompts from {PROMPTS_PATH}, active: {prompts['active_prompt']}")
    return prompts


def get_active_prompt() -> dict:
    """
    Returns the active prompt config including version string.

    Returns dict with keys: key, version, description, system
    """
    prompts = load_prompts()
    active_key = prompts["active_prompt"]

    if active_key not in prompts:
        raise ValueError(
            f"Active prompt '{active_key}' not found in prompts.yaml. "
            f"Available: {[k for k in prompts if k != 'active_prompt']}"
        )

    return {
        "key": active_key,
        **prompts[active_key],
    }


def reload_prompts() -> None:
    """Clear the prompt cache to force re-reading from disk."""
    load_prompts.cache_clear()
    logger.info("Prompt cache cleared — will reload on next access")
