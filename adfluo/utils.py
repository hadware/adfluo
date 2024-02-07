import logging
import sys
import typing
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("adfluo")


@dataclass
class ExtractionPolicy:
    skip_errors: bool = False
    no_cache: bool = False

