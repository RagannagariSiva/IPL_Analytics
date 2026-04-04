"""
utils/helpers.py
=================
Shared utility helpers used across the project.
"""

import os
import json
import hashlib
import time
from typing import Any, Dict, Optional
from pathlib import Path
from functools import wraps
from loguru import logger


def timeit(func):
    """Decorator: log execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns default on zero denominator."""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, value))


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> Dict:
    """Load a JSON file, return empty dict if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(data: Dict, path: str, indent: int = 2):
    """Save a dict to JSON file."""
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def file_checksum(path: str) -> str:
    """Compute SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def format_number(n: float, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    if n >= 1_000:
        return f"{n/1_000:.{decimals}f}K"
    return f"{n:.{decimals}f}"


def team_short_name(name: str) -> str:
    """Return 2-3 letter abbreviation for a team name."""
    abbreviations = {
        "Mumbai Indians":              "MI",
        "Chennai Super Kings":         "CSK",
        "Royal Challengers Bangalore": "RCB",
        "Kolkata Knight Riders":       "KKR",
        "Sunrisers Hyderabad":         "SRH",
        "Delhi Capitals":              "DC",
        "Punjab Kings":                "PBKS",
        "Rajasthan Royals":            "RR",
        "Gujarat Titans":              "GT",
        "Lucknow Super Giants":        "LSG",
        "Rising Pune Supergiants":     "RPS",
        "Pune Warriors India":         "PWI",
        "Deccan Chargers":             "DC2",
        "Kochi Tuskers Kerala":        "KTK",
    }
    return abbreviations.get(name, name[:3].upper())


def team_primary_color(name: str) -> str:
    """Return hex colour for a team's primary colour."""
    colors = {
        "Mumbai Indians":              "#004BA0",
        "Chennai Super Kings":         "#FDB913",
        "Royal Challengers Bangalore": "#EC1C24",
        "Kolkata Knight Riders":       "#3A225D",
        "Sunrisers Hyderabad":         "#FF822A",
        "Delhi Capitals":              "#0078BC",
        "Punjab Kings":                "#DCDDDE",
        "Rajasthan Royals":            "#2D4DA0",
        "Gujarat Titans":              "#1C1C1C",
        "Lucknow Super Giants":        "#A0E6FF",
    }
    return colors.get(name, "#e53935")


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent
