# Dependencies
import time
from typing import List
from pathlib import Path
from utils.logger import get_logger
from config.settings import settings
from config.schemas import EvidenceResult
from concurrent.futures import TimeoutError
from concurrent.futures import as_completed
from config.constants import EvidenceStrength
from config.constants import EvidenceDirection
from concurrent.futures import ThreadPoolExecutor
from config.constants import EVIDENCE_STRENGTH_ORDER
from evidence_analyzers.exif_analyzer import ExifAnalyzer
from evidence_analyzers.watermark_analyzer import WatermarkAnalyzer


# Setup Logging
logger = get_logger(__name__)


class EvidenceAggregator:
    """
    Tier-2 Evidence Orchestrator

    Responsibilities:
    -----------------
    - Execute all evidence analyzers safely
    - Collect declarative evidence only (no inference)
    - Deduplicate overlapping findings
    - Rank evidence by authority & reliability
    - Remain forward-compatible with new evidence sources
    """
    def __init__(self):
        """
        Initialize all Tier-2 analyzers

        NOTE:
        -----
        Any new analyzer (C2PA, camera fingerprinting, sensor PRNU) must be added here explicitly
        """
        self.exif_analyzer      = ExifAnalyzer()
        self.watermark_analyzer = WatermarkAnalyzer()

        self._analyzers         = (self.exif_analyzer,
                                   self.watermark_analyzer,
                                  )

        logger.info("EvidenceAggregator initialized with analyzers: "
                    f"{[a.__class__.__name__ for a in self._analyzers]}")


    def analyze(self, image_path: Path) -> List[EvidenceResult]:
        """
        Run Tier-2 evidence extraction pipeline

        Arguments:
        ----------
            image_path   {Path}  : Path to image file

        Returns:
        --------
                { list }         : Ordered, deduplicated evidence
        """
        # Small, bounded executor for Tier-2 (I/O oriented)
        max_workers        = min(len(self._analyzers), settings.EVIDENCE_WORKERS or 2)

        logger.info(f"Starting Tier-2 evidence analysis: {image_path}")

        evidence_collected = list()


        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = {executor.submit(analyzer.analyze, image_path = image_path): {"analyzer": analyzer, "start": time.time()} for analyzer in self._analyzers}
           
            for future in as_completed(futures):
                meta     = futures[future]
                analyzer = meta["analyzer"]
                start    = meta["start"]

                try:
                    results = future.result(timeout = settings.EVIDENCE_TIMEOUT)

                    logger.debug(f"{analyzer.__class__.__name__} completed in {time.time()-start:.2f}s")

                    if results:
                        evidence_collected.extend(results)
                        logger.debug(f"{analyzer.__class__.__name__} returned {len(results)} evidence items")
                    
                    else:
                        logger.debug(f"{analyzer.__class__.__name__} returned no evidence")

                except TimeoutError:
                    logger.warning(f"{analyzer.__class__.__name__} timed out")

                except Exception as e:
                    logger.error(f"{analyzer.__class__.__name__} failed: {e}")

        if not evidence_collected:
            logger.info("No Tier-2 evidence detected")
            return []

        # Normalize, deduplicate & rank
        evidence = self._deduplicate(evidence = evidence_collected)
        evidence = self._rank_evidence(evidence = evidence)

        logger.info(f"Tier-2 evidence finalized: {len(evidence)} items")
        
        return evidence


    def _deduplicate(self, evidence: List[EvidenceResult]) -> List[EvidenceResult]:
        """
        Deduplicate evidence items

        Strategy:
        ---------
        - Same analyzer
        - Same semantic finding
        - Same direction

        Keeps the strongest / highest confidence instance
        """
        unique_map = dict()

        for item in evidence:
            key = (item.analyzer, item.finding, item.direction)

            if key not in unique_map:
                unique_map[key] = item
                continue

            existing               = unique_map[key]
            existing_strength_rank = self._strength_rank(strength = existing.strength)
            item_strength_rank     = self._strength_rank(strength = item.strength)

            # Prefer stronger evidence
            if  (item_strength_rank > existing_strength_rank):
                unique_map[key] = item
                continue

            # Prefer higher confidence if strength equal
            if (item_strength_rank == existing_strength_rank):
                if (item.confidence or 0.0) > (existing.confidence or 0.0):
                    unique_map[key] = item

        deduped = list(unique_map.values())

        logger.debug(f"Deduplicated evidence: {len(evidence)} → {len(deduped)}")

        return deduped


    def _rank_evidence(self, evidence: List[EvidenceResult]) -> List[EvidenceResult]:
        """
        Rank evidence by authority

        Ranking precedence:
        -------------------
        1. Direction (AI > AUTHENTIC > INDETERMINATE)
        2. Strength  (CONCLUSIVE > STRONG > MODERATE > WEAK)
        3. Confidence (higher wins)
        """
        def priority(e: EvidenceResult) -> tuple:
            return (self._direction_rank(direction = e.direction), 
                    self._strength_rank(strength = e.strength),
                    e.confidence or 0.0,
                   )

        ranked = sorted(evidence, key = priority, reverse = True)

        logger.debug("Evidence ranking completed")
        
        return ranked


    @staticmethod
    def _direction_rank(direction: EvidenceDirection) -> int:
        """
        Evidence direction priority
        """
        return {EvidenceDirection.AI_GENERATED  : 3,
                EvidenceDirection.AUTHENTIC     : 2,
                EvidenceDirection.INDETERMINATE : 1,
               }.get(direction, 0)


    @staticmethod
    def _strength_rank(strength: EvidenceStrength) -> int:
        """
        Evidence strength priority
        """
        return EVIDENCE_STRENGTH_ORDER.get(strength, 0)