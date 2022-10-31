# standard libaries
import logging
from dataclasses import dataclass, field
from typing import List


@dataclass
class Word:
    """Class for a word and tags"""

    sentence_num: int
    word: str
    start_idx: int
    end_idx: int
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.tags.append("O")


def configure_logging() -> logging.Logger:
    """Make logger"""
    logger = logging.getLogger()
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    return logger
