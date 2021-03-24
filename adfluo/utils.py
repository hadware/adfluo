import logging

from dataclasses import dataclass

logger = logging.getLogger("mekhane")


@dataclass
class ExtractionPolicy:
    skip_errors: bool = False
    no_cache: bool = False


extraction_policy = ExtractionPolicy()
