"""
Data models for dynamic text pool system.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class PoolItem:
    """Represents an item in the text pool."""
    id: str
    text: str
    timestamp: str
    embedding: List[float]
    score: float


