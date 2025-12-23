# Dependencies
import pandas as pd
from typing import Dict
from typing import List
from utils.logger import get_logger
from config.constants import MetricType
from config.schemas import MetricResult
from config.constants import EvidenceType
from config.constants import SignalStatus
from config.schemas import AnalysisResult
from config.schemas import EvidenceResult
from config.constants import FinalDecision
from config.constants import EvidenceStrength
from config.constants import EvidenceDirection
from config.constants import SIGNAL_THRESHOLDS


# Setup Logging
logger = get_logger(__name__)


class DetailedResultMaker:
    """
    Extract and format detailed analysis results for reporting
    
    Purpose:
    --------
    - Extracts ALL data from AnalysisResult (metrics + evidence + decision)
    - Formats data into unified dictionaries/DataFrames
    - Provides structured data for reporters (JSON/CSV/PDF)
    - NO re-computation - pure data extraction and formatting
    
    Data Sources:
    -------------
    1. Final Decision (from DecisionPolicy)
    2. Evidence Results (from EvidenceAggregator)
    3. Metric Results (from SignalAggregator)
    4. Metadata (timestamps, processing info)
    
    Output Formats:
    ---------------
    1. Structured dictionaries for reporters
    2. Pandas DataFrames for tabular reports
    3. Hierarchical JSON-ready structures
    """
    def __init__(self, signal_thresholds: dict | None = None):
        """
        Initialize Detailed Result Maker
        """
        self.metric_display_names   = {MetricType.GRADIENT  : "Gradient-Field PCA",
                                       MetricType.FREQUENCY : "Frequency Domain (FFT)",
                                       MetricType.NOISE     : "Noise Pattern Analysis",
                                       MetricType.TEXTURE   : "Texture Statistics",
                                       MetricType.COLOR     : "Color Distribution",
                                      }

        self.evidence_display_names = {EvidenceType.EXIF      : "EXIF Metadata",
                                       EvidenceType.WATERMARK : "Watermark Detection",
                                      }

        self.decision_labels        = {FinalDecision.CONFIRMED_AI_GENERATED : "🔴 CONFIRMED AI GENERATED",
                                       FinalDecision.SUSPICIOUS_AI_LIKELY   : "🟠 SUSPICIOUS - AI LIKELY",
                                       FinalDecision.AUTHENTIC_BUT_REVIEW   : "🟡 AUTHENTIC BUT REVIEW",
                                       FinalDecision.MOSTLY_AUTHENTIC       : "🟢 MOSTLY AUTHENTIC",
                                      }

        self.signal_thresholds      = signal_thresholds or SIGNAL_THRESHOLDS
        
        logger.debug("DetailedResultMaker initialized")
    

    def extract_detailed_results(self, analysis_result: AnalysisResult) -> Dict:
        """
        Extract ALL detailed results from AnalysisResult into unified dictionary
        
        This is the MAIN extraction method - reporters call this!
        
        Arguments:
        ----------
            analysis_result { AnalysisResult } : Complete analysis result
        
        Returns:
        --------
                   { dict }                    : Comprehensive detailed results containing:
                                                 - final_decision (from DecisionPolicy)
                                                 - evidence_summary (from EvidenceAggregator)
                                                 - evidence_detailed (all evidence items)
                                                 - overall_summary (basic info)
                                                 - metrics_detailed (all metric results)
                                                 - metadata (stats and counts)
        """
        logger.debug(f"Extracting detailed results for: {analysis_result.filename}")
        
        detailed = {"filename"          : analysis_result.filename,
                    "final_decision"    : self._extract_final_decision(analysis_result),
                    "evidence_summary"  : self._extract_evidence_summary(analysis_result),
                    "evidence_detailed" : self._extract_all_evidence(analysis_result),
                    "overall_summary"   : self._extract_overall_summary(analysis_result),
                    "metrics_detailed"  : self._extract_all_metrics(analysis_result),
                    "metadata"          : self._extract_metadata(analysis_result),
                   }
        
        logger.debug(f"Extracted {len(detailed['evidence_detailed'])} evidence items, {len(detailed['metrics_detailed'])} metric details")
        
        return detailed
    

    def create_detailed_table(self, analysis_result: AnalysisResult) -> pd.DataFrame:
        """
        Create detailed table as DataFrame (for CSV export)
        
        Includes: Decision + Evidence + Metrics in hierarchical order
        
        Arguments:
        ----------
            analysis_result { AnalysisResult } : Complete analysis result
        
        Returns:
        --------
                      { DataFrame }            : Tabular detailed results
        """
        rows = list()
        
        # Final Decision (if available)
        if analysis_result.final_decision:
            decision_row = {"Type"        : "FINAL DECISION",
                            "Name"        : self.decision_labels.get(analysis_result.final_decision, analysis_result.final_decision.value),
                            "Score"       : "N/A",
                            "Confidence"  : f"{analysis_result.confidence}%",
                            "Status"      : analysis_result.final_decision.value.upper(),
                            "Explanation" : analysis_result.decision_explanation or "See evidence and metrics below",
                           }

            rows.append(decision_row)
        
        # Evidence (if any)
        if analysis_result.evidence:
            for evidence in analysis_result.evidence:
                source_key   = evidence.source.value if hasattr(evidence.source, "value") else str(evidence.source)

                evidence_row = {"Type"        : "EVIDENCE",
                                "Name"        : f"{self.evidence_display_names.get(source_key, source_key)} - {evidence.analyzer}",
                                "Score"       : f"{evidence.confidence:.2f}" if evidence.confidence is not None else "N/A",
                                "Confidence"  : f"{int(evidence.confidence * 100)}%" if evidence.confidence is not None else "N/A",
                                "Status"      : self._evidence_to_status_label(evidence),
                                "Explanation" : evidence.finding,
                               }

                rows.append(evidence_row)
        
        # Metrics
        for metric_type, metric_result in analysis_result.metric_results.items():
            display_name = self.metric_display_names.get(metric_type, metric_type.value)
            
            metric_row   = {"Type"       : "METRIC",
                            "Name"       : display_name,
                            "Score"      : round(metric_result.score, 3),
                            "Confidence" : f"{round(metric_result.confidence * 100)}%" if metric_result.confidence is not None else "N/A",
                            "Status"     : self._score_to_status(metric_result.score),
                           }
            
            # Add key details
            details      = self._extract_key_details(metric_type   = metric_type,
                                                     metric_result = metric_result,
                                                    )
            metric_row.update(details)
            
            rows.append(metric_row)
        
        dataframe = pd.DataFrame(data=rows)
        
        logger.debug(f"Created detailed table with {len(dataframe)} rows")
        
        return dataframe


    def _extract_final_decision(self, analysis_result: AnalysisResult) -> Dict:
        """
        Extract final decision information from DecisionPolicy
        """
        if not analysis_result.final_decision:
            return {"decision"    : None,
                    "label"       : "⚪ No Decision",
                    "explanation" : "Decision policy not applied",
                    "confidence"  : 0,
                    "based_on"    : "Unknown",
                   }

        final_decision = {"decision"    : analysis_result.final_decision.value,
                          "label"       : self.decision_labels.get(analysis_result.final_decision, analysis_result.final_decision.value),
                          "explanation" : analysis_result.decision_explanation or "No explanation provided",
                          "confidence"  : analysis_result.confidence,
                          "based_on"    : self._determine_decision_basis(analysis_result),
                         } 
        
        return final_decision


    def _determine_decision_basis(self, analysis_result: AnalysisResult) -> str:
        """
        Determine what the decision was based on
        """
        if not analysis_result.evidence:
            return "Statistical metrics only"
        
        # Check for strong evidence
        strong_evidence = [item for item in analysis_result.evidence if item.strength in (EvidenceStrength.STRONG, EvidenceStrength.CONCLUSIVE)]
        
        if strong_evidence:
            evidence_types = {item.source.value if hasattr(item.source, "value") else str(item.source) for item in strong_evidence}
            return f"Strong evidence (Tier 2): {', '.join(evidence_types)}"
        
        return "Combination of evidence and metrics (Tier 2 + Tier 1)"


    def _extract_evidence_summary(self, analysis_result: AnalysisResult) -> Dict:
        """
        Extract high-level evidence summary
        """
        if not analysis_result.evidence:
            return {"total_evidence"      : 0,
                    "ai_evidence_count"   : 0,
                    "auth_evidence_count" : 0,
                    "strongest_evidence"  : None,
                   }
        
        ai_evidence   = [item for item in analysis_result.evidence if (item.direction == EvidenceDirection.AI_GENERATED)]
        auth_evidence = [item for item in analysis_result.evidence if (item.direction == EvidenceDirection.AUTHENTIC)]
        
        # Find strongest evidence
        strongest     = max(analysis_result.evidence,
                            key = lambda item: (self._strength_to_rank(item.strength), item.confidence or 0.0)
                           )
        
        return {"total_evidence"      : len(analysis_result.evidence),
                "ai_evidence_count"   : len(ai_evidence),
                "auth_evidence_count" : len(auth_evidence),
                "strongest_evidence"  : {"source"     : strongest.source.value,
                                         "direction"  : strongest.direction.value,
                                         "strength"   : strongest.strength.value,
                                         "finding"    : strongest.finding,
                                         "confidence" : strongest.confidence,
                                        },
               }


    def _extract_all_evidence(self, analysis_result: AnalysisResult) -> List[Dict]:
        """
        Extract detailed information for all evidence items
        """
        if not analysis_result.evidence:
            return []
        
        evidence_detailed = list()
        
        for evidence in analysis_result.evidence:
            timestamp         = getattr(evidence, "timestamp", None)

            evidence_detail = {"source"       : evidence.source.value,
                               "display_name" : self.evidence_display_names.get(evidence.source.value if hasattr(evidence.source, "value") else str(evidence.source), str(evidence.source)),
                               "finding"      : evidence.finding,
                               "direction"    : evidence.direction.value,
                               "strength"     : evidence.strength.value,
                               "confidence"   : evidence.confidence,
                               "analyzer"     : evidence.analyzer,
                               "details"      : evidence.details,
                               "status_label" : self._evidence_to_status_label(evidence),
                               "timestamp"    : timestamp.isoformat() if timestamp else None,
                              }

            evidence_detailed.append(evidence_detail)
        
        return evidence_detailed


    def _evidence_to_status_label(self, evidence: EvidenceResult) -> str:
        """
        Convert evidence to human-readable status label
        """
        if (evidence.direction == EvidenceDirection.AI_GENERATED):
            if (evidence.strength == EvidenceStrength.CONCLUSIVE):
                return "🔴 CONCLUSIVE AI"

            elif (evidence.strength == EvidenceStrength.STRONG):
                return "🔴 STRONG AI"

            elif (evidence.strength == EvidenceStrength.MODERATE):
                return "🟠 MODERATE AI"

            else:
                return "🟡 WEAK AI"
        
        elif (evidence.direction == EvidenceDirection.AUTHENTIC):
            if (evidence.strength in (EvidenceStrength.STRONG, EvidenceStrength.CONCLUSIVE)):
                return "🟢 STRONG AUTHENTIC"

            elif (evidence.strength == EvidenceStrength.MODERATE):
                return "🟢 MODERATE AUTHENTIC"

            else:
                return "🟡 WEAK AUTHENTIC"
        
        else:  
            # INDETERMINATE
            return "⚪ INDETERMINATE"


    def _strength_to_rank(self, strength: EvidenceStrength) -> int:
        """
        Convert strength to numeric rank for sorting
        """
        return {EvidenceStrength.CONCLUSIVE : 4,
                EvidenceStrength.STRONG     : 3,
                EvidenceStrength.MODERATE   : 2,
                EvidenceStrength.WEAK       : 1,
               }.get(strength, 0)


    def _extract_overall_summary(self, analysis_result: AnalysisResult) -> Dict:
        """
        Extract overall summary information
        """
        timestamp = getattr(analysis_result, "timestamp", None)

        return {"filename"        : analysis_result.filename,
                "status"          : analysis_result.status.value,
                "overall_score"   : round(analysis_result.overall_score, 3),
                "confidence"      : analysis_result.confidence,
                "processing_time" : round(analysis_result.processing_time, 2),
                "image_size"      : f"{analysis_result.image_size[0]}×{analysis_result.image_size[1]}",
                "timestamp"       : timestamp.isoformat() if timestamp else None,
               }
    

    def _extract_all_metrics(self, analysis_result: AnalysisResult) -> List[Dict]:
        """
        Extract detailed information for all metrics
        """
        metrics_detailed = list()
        
        for metric_type, metric_result in analysis_result.metric_results.items():
            metric_detail = {"metric_type"    : metric_type.value,
                             "display_name"   : self.metric_display_names.get(metric_type, metric_type.value),
                             "score"          : round(metric_result.score, 3),
                             "confidence"     : round(metric_result.confidence, 3) if metric_result.confidence is not None else None,
                             "status"         : self._score_to_status(metric_result.score),
                             "details"        : metric_result.details or {},
                             "interpretation" : self._interpret_metric(metric_type, metric_result),
                             "key_findings"   : self.extract_key_findings(metric_type, metric_result),
                            }

            metrics_detailed.append(metric_detail)
        
        # Sort by score (highest first)
        metrics_detailed.sort(key     = lambda x: x['score'], 
                              reverse = True,
                             )
        
        return metrics_detailed
    

    def _extract_metadata(self, analysis_result: AnalysisResult) -> Dict:
        """
        Extract processing metadata and statistics
        """
        metadata = {"total_metrics"   : len(analysis_result.metric_results),
                    "flagged_metrics" : sum(1 for s in analysis_result.signals if s.status == SignalStatus.FLAGGED),
                    "warning_metrics" : sum(1 for s in analysis_result.signals if s.status == SignalStatus.WARNING),
                    "passed_metrics"  : sum(1 for s in analysis_result.signals if s.status == SignalStatus.PASSED),
                    "avg_confidence"  : self._calculate_avg_confidence(analysis_result),
                   }
                
        # Evidence stats (if available)
        if analysis_result.evidence:
            metadata["total_evidence"]  = len(analysis_result.evidence)
            metadata["ai_evidence"]     = sum(1 for e in analysis_result.evidence if e.direction == EvidenceDirection.AI_GENERATED)
            metadata["auth_evidence"]   = sum(1 for e in analysis_result.evidence if e.direction == EvidenceDirection.AUTHENTIC)
            metadata["strong_evidence"] = sum(1 for e in analysis_result.evidence if e.strength in (EvidenceStrength.STRONG, EvidenceStrength.CONCLUSIVE))

        else:
            metadata["total_evidence"]  = 0
            metadata["ai_evidence"]     = 0
            metadata["auth_evidence"]   = 0
            metadata["strong_evidence"] = 0
        
        # Decision info
        metadata["has_final_decision"] = analysis_result.final_decision is not None
        metadata["decision_value"]     = analysis_result.final_decision.value if analysis_result.final_decision else None
        
        return metadata


    def _extract_key_details(self, metric_type: MetricType, metric_result: MetricResult) -> Dict:
        """
        Extract key details specific to each metric type
        """
        details = metric_result.details or {}
        
        if (metric_type == MetricType.GRADIENT):
            return {"Eigenvalue_Ratio" : details.get('eigenvalue_ratio', 'N/A'),
                    "Vectors_Sampled"  : details.get('gradient_vectors_sampled', 'N/A'),
                   }
        
        elif (metric_type == MetricType.FREQUENCY):
            return {"HF_Ratio"      : details.get('hf_ratio', 'N/A'),
                    "HF_Anomaly"    : details.get('hf_anomaly', 'N/A'),
                    "Spectrum_Bins" : details.get('spectrum_bins', 'N/A'),
                   }
        
        elif (metric_type == MetricType.NOISE):
            return {"Mean_Noise"    : details.get('mean_noise', 'N/A'),
                    "CV"            : details.get('cv', 'N/A'),
                    "Patches_Valid" : details.get('patches_valid', 'N/A'),
                   }
        
        elif (metric_type == MetricType.TEXTURE):
            return {"Smooth_Ratio"  : details.get('smooth_ratio', 'N/A'),
                    "Contrast_Mean" : details.get('contrast_mean', 'N/A'),
                    "Patches_Used"  : details.get('patches_used', 'N/A'),
                   }
        
        elif (metric_type == MetricType.COLOR):
            sat_stats = details.get('saturation_stats', {})

            return {"Mean_Saturation" : sat_stats.get('mean_saturation', 'N/A'),
                    "High_Sat_Ratio"  : sat_stats.get('high_sat_ratio', 'N/A'),
                   }
        
        return {}
    

    def _interpret_metric(self, metric_type: MetricType, metric_result: MetricResult) -> str:
        """
        Provide human-readable interpretation of metric result
        """
        details = metric_result.details or {}
        
        if (metric_type == MetricType.GRADIENT):
            eig_ratio = details.get('eigenvalue_ratio')
            
            if eig_ratio:
                return f"Eigenvalue ratio of {eig_ratio:.3f} ({'high' if (eig_ratio > 0.85) else 'low'} alignment)"
            
            return "Gradient structure analysis"
        
        elif( metric_type == MetricType.FREQUENCY):
            hf_ratio = details.get('hf_ratio')

            if hf_ratio:
                return f"High-freq ratio: {hf_ratio:.3f} ({'elevated' if (hf_ratio > 0.35) else 'low' if (hf_ratio < 0.08) else 'normal'})"
            
            return "Frequency spectrum analysis"
        
        elif (metric_type == MetricType.NOISE):
            mean_noise = details.get('mean_noise')

            if mean_noise:
                return f"Mean noise: {mean_noise:.2f} ({'low' if (mean_noise < 1.5) else 'normal'})"
            
            return "Noise pattern analysis"
        
        elif (metric_type == MetricType.TEXTURE):
            smooth_ratio = details.get('smooth_ratio')

            if smooth_ratio is not None:
                return f"Smooth regions: {smooth_ratio:.1%} ({'excessive' if (smooth_ratio > 0.4) else 'normal'})"
            
            return "Texture variation analysis"
        
        elif (metric_type == MetricType.COLOR):
            sat_stats = details.get('saturation_stats', {})
            mean_sat  = sat_stats.get('mean_saturation')
            
            if mean_sat:
                return f"Mean saturation: {mean_sat:.2f} ({'high' if (mean_sat > 0.65) else 'normal'})"
            
            return "Color distribution analysis"
        
        return "Analysis complete"


    def extract_key_findings(self, metric_type: MetricType, metric_result: MetricResult) -> List[str]:
        """
        Extract human-readable key forensic findings for reporters
        """
        findings = list()
        details  = metric_result.details or {}
        
        if (metric_type == MetricType.GRADIENT):
            eig_ratio = details.get('eigenvalue_ratio')

            if eig_ratio:
                findings.append(f"Eigenvalue ratio: {eig_ratio:.3f}")
            
            vectors = details.get('gradient_vectors_sampled')
            
            if vectors:
                findings.append(f"Analyzed {vectors} gradient vectors")
        
        elif (metric_type == MetricType.FREQUENCY):
            hf_ratio = details.get('hf_ratio')

            if hf_ratio:
                findings.append(f"High-frequency ratio: {hf_ratio:.3f}")
            
            roughness = details.get('roughness')
            
            if roughness:
                findings.append(f"Spectral roughness: {roughness:.3f}")
        
        elif (metric_type == MetricType.NOISE):
            mean_noise = details.get('mean_noise')

            if mean_noise:
                findings.append(f"Mean noise level: {mean_noise:.2f}")
            
            cv = details.get('cv')
            
            if cv:
                findings.append(f"Coefficient of variation: {cv:.3f}")
        
        elif (metric_type == MetricType.TEXTURE):
            smooth_ratio = details.get('smooth_ratio')

            if smooth_ratio:
                findings.append(f"Smooth patches: {smooth_ratio:.1%}")
            
            contrast_mean = details.get('contrast_mean')
            
            if contrast_mean:
                findings.append(f"Average contrast: {contrast_mean:.2f}")
        
        elif (metric_type == MetricType.COLOR):
            sat_stats = details.get('saturation_stats', {})
            mean_sat  = sat_stats.get('mean_saturation')
            
            if mean_sat:
                findings.append(f"Mean saturation: {mean_sat:.2f}")
            
            high_sat = sat_stats.get('high_sat_ratio')
            
            if high_sat:
                findings.append(f"High saturation pixels: {high_sat:.1%}")
        
        return findings if findings else ["Analysis complete"]


    def _score_to_status(self, score: float) -> str:
        """
        Convert score to status label
        """
        if (score >= self.signal_thresholds[SignalStatus.FLAGGED]):
            return "FLAGGED"

        elif (score >= self.signal_thresholds[SignalStatus.WARNING]):
            return "WARNING"

        else:
            return "PASSED"


    def _calculate_avg_confidence(self, analysis_result: AnalysisResult) -> float:
        """
        Calculate average confidence across all metrics
        """
        confidences = [mr.confidence for mr in analysis_result.metric_results.values() if mr.confidence is not None]
        
        return round(sum(confidences) / len(confidences), 3) if confidences else 0.0