"""
Configuration for Multi-Page Document Layout Analysis Pipeline.

This module centralizes all configurable parameters so that
experiments can be reproduced and tweaked from a single place.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "example_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ── LLM / OpenRouter ────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3.1-pro-preview"

# ── Phase 1: Category Discovery ─────────────────────────────────
# Sliding window size: how many consecutive pages the MLLM sees at once.
CONTEXT_WINDOW_SIZE = 3

# ── Phase 2: Physical Analysis ──────────────────────────────────
# How many pages to send per analysis call.
# Each window is analyzed independently; overlapping pages are deduplicated.
PHASE2_WINDOW_SIZE = 3

# Phase 2 outputs many elements → needs a larger token budget.
PHASE2_MAX_TOKENS = 16384

# ── Shared generation parameters ────────────────────────────────
TEMPERATURE = 0.0
MAX_TOKENS = 8192
