import logging
from dataclasses import dataclass

logger = logging.getLogger("adfluo")


@dataclass
class ExtractionPolicy:
    skip_errors: bool = False
    no_cache: bool = False

