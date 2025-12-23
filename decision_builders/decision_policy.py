# Dependencies
from typing import List
from utils.logger import get_logger
from config.schemas import AnalysisResult
from config.schemas import EvidenceResult
from config.constants import FinalDecision
from config.constants import DetectionStatus
from config.constants import EvidenceStrength
from config.constants import EvidenceDirection
from config.constants import MIN_EVIDENCE_CONFIDENCE
from config.constants import EVIDENCE_STRENGTH_ORDER


# Setup Logging
logger = get_logger(__name__)


class DecisionPolicy:
    """
    Evidence-first decision policy

    Responsibilities:
    -----------------
    - Apply authoritative rules over Tier-1 metrics
    - Resolve conflicting evidence safely
    - Produce a final, explainable decision
    - Never perform probabilistic inference
    """
    def apply(self, analysis: AnalysisResult) -> AnalysisResult:
        """
        Apply final decision policy on analysis result
        """
        try:
            evidence = analysis.evidence or []
            
            if not evidence:
                analysis.final_decision       = self._decide_from_metrics(status = analysis.status)
                analysis.decision_explanation = ("No declarative evidence detected. Final decision is based on metric analysis.")
                
                return analysis

            logger.info(f"Applying decision policy on {len(evidence)} evidence items")

            decision, explanation         = self._decide_from_evidence(evidence        = evidence,
                                                                       fallback_status = analysis.status,
                                                                      )

            analysis.final_decision       = decision
            analysis.decision_explanation = explanation

            logger.info(f"Final decision resolved: {decision.value}")

            return analysis

        except Exception as e:
            # Decision policy must NEVER break pipeline
            logger.error(f"Decision policy failed: {e}")
            analysis.final_decision = FinalDecision.AUTHENTIC_BUT_REVIEW

            analysis.decision_explanation = ("Decision policy encountered an internal error. Image requires manual review.")
            return analysis

    
    def _decide_from_evidence(self, evidence: List[EvidenceResult], fallback_status: DetectionStatus) -> tuple[FinalDecision, str]:
        """
        Resolve decision from Tier-2 evidence with explanation
        """
        # Storage for evidences
        ai_evidence         = list()
        authentic_evidence  = list()

        indeterminate_count = sum(1 for e in evidence if (e.direction == EvidenceDirection.INDETERMINATE))

        for item in evidence:
            if (item.direction == EvidenceDirection.AI_GENERATED):
                ai_evidence.append(item)
            
            elif (item.direction == EvidenceDirection.AUTHENTIC):
                authentic_evidence.append(item)

            else:
                logger.debug(f"Indeterminate evidence detected and excluded from decisive rules: {item.finding} ({item.analyzer})")
                

        strongest_ai        = self._strongest(evidence = ai_evidence)
        strongest_authentic = self._strongest(evidence = authentic_evidence)

        # Rule 1: CONCLUSIVE AI
        if (strongest_ai and (strongest_ai.strength == EvidenceStrength.CONCLUSIVE) and ((strongest_ai.confidence or 0.0) >= MIN_EVIDENCE_CONFIDENCE)):
            return (FinalDecision.CONFIRMED_AI_GENERATED,
                    f"Conclusive evidence detected ({strongest_ai.finding}). This evidence provides cryptographic or authoritative proof of AI generation.",
                   )

        # Rule 2: Strong AI evidence 
        if (strongest_ai and (strongest_ai.strength == EvidenceStrength.STRONG) and ((strongest_ai.confidence or 0.0) >= MIN_EVIDENCE_CONFIDENCE)):
            if (strongest_authentic and (strongest_authentic.strength in (EvidenceStrength.STRONG, EvidenceStrength.CONCLUSIVE))):
                return (FinalDecision.AUTHENTIC_BUT_REVIEW,
                        "Strong evidence exists for both AI generation and authenticity. Conflicting high-authority signals require human review.",
                       )

            return (FinalDecision.SUSPICIOUS_AI_LIKELY,
                    f"Strong AI-related evidence detected ({strongest_ai.finding}). This evidence overrides metric-based assessment.")

        # Rule 3: Strong authentic evidence
        if (strongest_authentic and (strongest_authentic.strength in (EvidenceStrength.STRONG, EvidenceStrength.CONCLUSIVE))):
            return (FinalDecision.MOSTLY_AUTHENTIC,
                    f"Strong authenticity evidence detected ({strongest_authentic.finding}). Image is assessed as mostly authentic."
                   )

        # Rule 4: Conflicting moderate evidence
        if (strongest_ai and strongest_authentic):
            return (FinalDecision.AUTHENTIC_BUT_REVIEW,
                    "Moderate evidence exists for both AI generation and authenticity. Evidence is inconclusive and requires manual review.")

        if (indeterminate_count >= 2):
            return (FinalDecision.AUTHENTIC_BUT_REVIEW,
                    "Multiple indeterminate evidence sources detected. Evidence is inconclusive and requires manual review."
                   )

        # Rule 5: Weak / indeterminate evidence
        decision = self._decide_from_metrics(status = fallback_status)
        return (decision,
                "No high-authority evidence detected. Final decision is derived from metric-based analysis.",
               )

    
    def _decide_from_metrics(self, status: DetectionStatus) -> FinalDecision:
        """
        Convert Tier-1 metric status into final decision
        """
        if (status == DetectionStatus.REVIEW_REQUIRED):
            return FinalDecision.SUSPICIOUS_AI_LIKELY

        return FinalDecision.MOSTLY_AUTHENTIC

    
    def _strongest(self, evidence: List[EvidenceResult]) -> EvidenceResult | None:
        """
        Return strongest evidence by strength, then confidence
        """
        if not evidence:
            return None

        return max(evidence,
                   key = lambda item: (EVIDENCE_STRENGTH_ORDER.get(item.strength, 0), item.confidence or 0.0),
                  )