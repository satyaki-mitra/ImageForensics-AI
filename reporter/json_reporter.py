# Dependencies
import json
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional
from datetime import datetime
from utils.logger import get_logger
from config.settings import settings
from config.schemas import AnalysisResult
from config.constants import FinalDecision
from utils.helpers import generate_unique_id
from config.schemas import BatchAnalysisResult
from features.detailed_result_maker import DetailedResultMaker


# Setup Logging
logger = get_logger(__name__)


class JSONReporter:
    """
    JSON report generator

    Guarantees:
    -----------
    - FinalDecision is authoritative
    - Metrics are informational only
    - Evidence-first interpretation
    - Audit-safe output
    """
    def __init__(self):
        """
        Initialize JSON Reporter
        """
        self.detailed_maker = DetailedResultMaker()

        logger.debug("JSONReporter initialized")
    

    def export_batch(self, batch_result: BatchAnalysisResult, output_dir: Optional[Path] = None, include_detailed: bool = True) -> Path:
        """
        Export batch analysis as JSON
        """
        output_dir  = output_dir or settings.REPORTS_DIR
        report_id   = generate_unique_id()
        filename    = f"batch_report_{report_id}.json"
        output_path = output_dir / filename

        output_dir.mkdir(parents = True, exist_ok = True)
        logger.info(f"Generating batch JSON: {filename}")

        try:
            data = self._build_batch_json(batch_result     = batch_result,
                                          include_detailed = include_detailed,
                                         )

            with open(output_path, 'w', encoding = 'utf-8') as f:
                json.dump(obj          = data,
                          fp           = f,
                          indent       = 4,
                          ensure_ascii = False,
                          default      = str,
                         )

            logger.info(f"Batch JSON generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate batch JSON: {e}")
            raise
    

    def export_single(self, result: AnalysisResult, output_dir: Optional[Path] = None, include_detailed: bool = True,
                     ) -> Path:
        """
        Export single image analysis as JSON
        """
        output_dir  = output_dir or settings.REPORTS_DIR
        report_id   = generate_unique_id()
        filename    = f"single_report_{report_id}.json"
        output_path = output_dir / filename

        output_dir.mkdir(parents = True, exist_ok = True)
        logger.info(f"Generating single image JSON: {filename}")

        try:
            data = self._build_single_json(result           = result,
                                           include_detailed = include_detailed,
                                          )

            with open(output_path, 'w', encoding = 'utf-8') as f:
                json.dump(obj          = data,
                          fp           = f,
                          indent       = 4,
                          ensure_ascii = False,
                          default      = str,
                         )

            logger.info(f"Single image JSON generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate single image JSON: {e}")
            raise
    

    def export_api_response(self, result: AnalysisResult) -> Dict:
        """
        Generate API-friendly JSON response
        """
        return {"success"   : True,
                "timestamp" : datetime.now().isoformat(),
                "version"   : settings.VERSION,
                "data"      : self._build_single_json(result           = result,
                                                      include_detailed = False,
                                                     ),
               }
    

    def _build_batch_json(self, batch_result: BatchAnalysisResult, include_detailed: bool) -> Dict:
        """
        Build complete batch JSON structure
        """
        return {"report_metadata" : self._build_metadata(report_type = "Batch Analysis",
                                                         timestamp   = batch_result.timestamp,
                                                        ),
                "batch_summary"   : self._build_batch_summary(batch_result),
                "results"         : [self._build_image_data(result, include_detailed) for result in batch_result.results],
               }
    

    def _build_single_json(self, result: AnalysisResult, include_detailed: bool) -> Dict:
        """
        Build single image JSON structure
        """
        return {"report_metadata" : self._build_metadata(report_type = "Single Image Analysis",
                                                         timestamp   = result.timestamp,
                                                        ),
                "analysis"        : self._build_image_data(result, include_detailed),
               }
    

    def _build_metadata(self, report_type: str, timestamp: datetime) -> Dict:
        """
        Build report metadata section
        """
        return {"report_type"    : report_type,
                "generated_at"   : timestamp.isoformat(),
                "generator"      : "AI Image Screener",
                "version"        : settings.VERSION,
                "format_version" : "1.0",
               }
    

    def _build_batch_summary(self, batch_result: BatchAnalysisResult) -> Dict:
        """
        Build batch summary (decision-aware)
        """
        summary = batch_result.summary or {}

        return {"total_images"          : batch_result.total_images,
                "processed"             : batch_result.processed,
                "failed"                : batch_result.failed,
                "success_rate"          : summary.get("success_rate", 0),
                "decision_distribution" : {key : summary.get(key, 0)
                                                 for key in [FinalDecision.CONFIRMED_AI_GENERATED.value,
                                                             FinalDecision.SUSPICIOUS_AI_LIKELY.value,
                                                             FinalDecision.AUTHENTIC_BUT_REVIEW.value,
                                                             FinalDecision.MOSTLY_AUTHENTIC.value,
                                                            ]
                                          },
                "total_processing_time" : round(batch_result.total_processing_time, 2),
               }
    

    def _build_image_data(self, result: AnalysisResult, include_detailed: bool) -> Dict:
        """
        Build complete image data structure (decision-first)
        """
        image_data = {"filename"   : result.filename,
                      "decision"   : {"value"        : result.final_decision.value if result.final_decision else None,
                                      "confidence"   : result.confidence,
                                      "explanation"  : result.decision_explanation,
                                     },
                      "overall"    : {"score"      : round(result.overall_score, 3),
                                      "note"       : "Statistical score (non-authoritative)",
                                     },
                      "image_info" : {"size"            : {"width"  : result.image_size[0],
                                                           "height" : result.image_size[1],
                                                          },
                                      "processing_time" : round(result.processing_time, 2),
                                      "timestamp"       : result.timestamp.isoformat(),
                                     },
                      "signals"    : self._build_signals_data(result),
                     }

        if include_detailed:
            image_data["forensics"]       = self._build_forensics_data(result)
            image_data["recommendations"] = self._build_recommendations(result)

        return image_data
    

    def _build_signals_data(self, result: AnalysisResult) -> List[Dict]:
        """
        Build Tier-1 signal data (informational)
        """
        signals = list()

        for signal in result.signals:
            metric_result = result.metric_results.get(signal.metric_type)

            signals.append({"metric_name" : signal.name,
                            "metric_type" : signal.metric_type.value,
                            "score"       : round(signal.score, 3),
                            "status"      : signal.status.value,
                            "confidence"  : round(metric_result.confidence, 3) if (metric_result and metric_result.confidence is not None) else None,
                            "explanation" : signal.explanation,
                           })

        return signals
    

    def _build_forensics_data(self, result: AnalysisResult) -> Dict:
        """
        Build forensic metric details
        """
        forensics = dict()

        for metric_type, metric_result in result.metric_results.items():
            forensics[metric_type.value] = {"display_name" : self.detailed_maker.metric_display_names.get(metric_type, metric_type.value),
                                            "score"        : round(metric_result.score, 3),
                                            "confidence"   : round(metric_result.confidence, 3) if metric_result.confidence is not None else None,
                                            "details"      : metric_result.details or {},
                                            "key_findings" : self.detailed_maker.extract_key_findings(metric_type, metric_result),
                                           }

        return forensics
    

    def _build_recommendations(self, result: AnalysisResult) -> Dict:
        """
        Build recommendations (decision-driven, not score-driven)
        """
        decision = result.final_decision

        if (decision == FinalDecision.CONFIRMED_AI_GENERATED):
            return {"action"     : "Block or flag image immediately",
                    "priority"   : "CRITICAL",
                    "next_steps" : ["Audit source", "Apply AI-content policy"],
                   }

        if (decision == FinalDecision.SUSPICIOUS_AI_LIKELY):
            return {"action"     : "Manual review required",
                    "priority"   : "HIGH",
                    "next_steps" : ["Human inspection", "Cross-check metadata"],
                   }

        if (decision == FinalDecision.AUTHENTIC_BUT_REVIEW):
            return {"action"     : "Optional human review",
                    "priority"   : "MEDIUM",
                    "next_steps" : ["Spot-check authenticity"],
                   }

        if (decision == FinalDecision.MOSTLY_AUTHENTIC):
            return {"action"     : "No action required",
                    "priority"   : "LOW",
                    "next_steps" : ["Proceed normally"],
                   }

        return {"action"     : "Decision unavailable",
                "priority"   : "UNKNOWN",
                "next_steps" : [],
               }