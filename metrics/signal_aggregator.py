# Dependencies
import os
import time
import numpy as np
from typing import List
from pathlib import Path
from types import MappingProxyType
from utils.logger import get_logger
from config.settings import settings
from config.schemas import MetricResult
from config.constants import MetricType
from config.constants import SignalStatus
from config.schemas import AnalysisResult
from config.schemas import DetectionSignal
from concurrent.futures import as_completed
from config.constants import DetectionStatus
from config.constants import SIGNAL_THRESHOLDS
from utils.image_processor import ImageProcessor
from config.constants import METRIC_EXPLANATIONS
from metrics.noise_analyzer import NoiseAnalyzer
from metrics.color_analyzer import ColorAnalyzer
from concurrent.futures import ThreadPoolExecutor
from metrics.texture_analyzer import TextureAnalyzer
from config.constants import SIGNAL_CONFIDENCE_PARAMS
from features.threshold_manager import ThresholdManager
from config.constants import IMAGE_RESIZE_MAX_DIMENSION
from metrics.frequency_analyzer import FrequencyAnalyzer
from metrics.gradient_field_pca import GradientFieldPCADetector


# Suppress NumPy warning 
np.seterr(divide  = 'ignore', 
          invalid = 'ignore',
         )

         
# Setup Logging
logger = get_logger(__name__)


class SignalAggregator:
    """
    Main detector that orchestrates all detection signals

    Combines multiple unsupervised metric signals:
    ----------------------------------------------
    1. Gradient-Field PCA
    2. Frequency Domain Analysis (FFT)
    3. Noise Pattern Analysis
    4. Texture Analysis
    5. Color Distribution Analysis

    Note: Each metric produces a suspicion score [0.0, 1.0] : scores are combined using weighted average to produce final assessment
    """
    def __init__(self, threshold_manager: ThresholdManager | None = None):
        """
        Initialize all detectors
        """
        logger.info("Initializing AI Image Detector")

        # Optional runtime threshold manager
        self.threshold_manager           = threshold_manager
        
        self.gradient_field_pca_detector = GradientFieldPCADetector()
        self.frequency_analyzer          = FrequencyAnalyzer()
        self.noise_analyzer              = NoiseAnalyzer()
        self.texture_analyzer            = TextureAnalyzer()
        self.color_analyzer              = ColorAnalyzer()
        self.image_processor             = ImageProcessor()

        # Create detector registry
        self.detector_registry          = MappingProxyType({MetricType.GRADIENT  : ("Gradient Field PCA", self.gradient_field_pca_detector),
                                                            MetricType.FREQUENCY : ("Frequency Analysis", self.frequency_analyzer),
                                                            MetricType.NOISE     : ("Noise Analysis", self.noise_analyzer),
                                                            MetricType.TEXTURE   : ("Texture Analysis", self.texture_analyzer),
                                                            MetricType.COLOR     : ("Color Analysis", self.color_analyzer),
                                                          })
        
        # Get metric weights either from runtime UI or default to settings
        self.weights                    = (self.threshold_manager.get_metric_weights() if self.threshold_manager else settings.get_metric_weights())
        
        # Initialize shared ThreadPoolExecutor (CPU-safe)
        max_workers                     = min(settings.METRIC_WORKERS or len(self.detector_registry), os.cpu_count() or 4)

        self.executor                   = ThreadPoolExecutor(max_workers = max_workers)
        
        logger.info(f"Metric weights: {self.weights}")


    def analyze_image(self, image_path: Path, filename: str, image_size: tuple) -> AnalysisResult:
        """
        Analyze single image for AI generation
        
        Arguments:
        ----------
            image_path { Path }  : Path to image file
            
            filename   { str }   : Original filename
            
            image_size { tuple } : (width, height) tuple
        
        Returns:
        --------
            { AnalysisResult }   : AnalysisResult with detection outcome
        """
        logger.info(f"Analyzing image: {filename}")
        
        start_time = time.time()
        
        try:
            # Load image
            image           = self.image_processor.load_image(file_path = image_path)
            
            # Resize if needed for performance
            image           = self.image_processor.resize_if_needed(image         = image, 
                                                                    max_dimension = IMAGE_RESIZE_MAX_DIMENSION,
                                                                   )
            
            # Run all detectors and get raw scores
            metric_results  = self._run_all_detectors(image = image)
            
            # Create signals from scores (aggregator's responsibility)
            signals         = self._create_signals_from_scores(metric_results = metric_results)
            
            # Aggregate results
            overall_score   = self._aggregate_scores(metric_results = metric_results)
            
            # Determine status
            status          = self._determine_status(overall_score = overall_score)

            # Calculate confidence 
            confidence      = self._calculate_confidence(metric_results = metric_results,
                                                         overall_score  = overall_score,
                                                        )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result          = AnalysisResult(filename        = filename,
                                             overall_score   = overall_score,
                                             status          = status,
                                             confidence      = confidence,
                                             signals         = signals,
                                             metric_results  = metric_results,
                                             processing_time = processing_time,
                                             image_size      = image_size,
                                            )
            
            logger.info(f"Analysis complete for {filename}: status={status.value}, score={overall_score:.3f}, time={processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {filename}: {e}")
            raise


    def _run_all_detectors(self, image: np.ndarray) -> dict[MetricType, MetricResult]:
        """
        Run all detection methods and collect raw scores
        
        Arguments:
        ----------
            image { np.ndarray } : RGB image array
        
        Returns:
        --------
                  { dict }       : Dictionary mapping MetricType to MetricResult
        """
        metric_results = dict()
        futures        = dict()
        
        # Submit all detectors
        for metric_type, (detector_name, detector) in self.detector_registry.items():

            futures[self.executor.submit(detector.detect, image = image)] = (metric_type, detector_name)
        
        # Collect results as they complete
        for future in as_completed(futures):
            metric_type, detector_name = futures[future]

            try:
                result                      = future.result(timeout = settings.METRIC_TIMEOUT)
                result.metric_type          = metric_type
                metric_results[metric_type] = result
                
                logger.debug(f"{detector_name} | {metric_type.value} | score={result.score:.3f} | confidence={result.confidence:.3f}")

            except Exception as e:
                logger.error(f"{detector_name} failed: {e}")

                # Same Failure Score by all metrics with same confidence 
                metric_results[metric_type] = MetricResult(metric_type = metric_type,
                                                           score       = settings.REVIEW_THRESHOLD,
                                                           confidence  = 0.0,
                                                           details     = {"error": "detector_failed"},
                                                          )
        
        return metric_results


    def _create_signals_from_scores(self, metric_results: dict) -> List[DetectionSignal]:
        """
        Convert MetricResults to DetectionSignals with status and explanations
        
        This is the aggregator's responsibility - metrics don't know about signals
        
        Arguments:
        ----------
            metric_results { dict }   : Dictionary mapping MetricType to float score
        
        Returns:
        --------
                    { list }          : List of complete detection signals
        """
        signals           = list()

        signal_thresholds = (self.threshold_manager.get_signal_thresholds() if self.threshold_manager else SIGNAL_THRESHOLDS)
        
        for metric_type, result in metric_results.items():
            # Extract score of the metric
            score = result.score

            # Determine status based on thresholds
            if (score >= signal_thresholds[SignalStatus.FLAGGED]):
                status   = SignalStatus.FLAGGED
                severity = 'high'
                
            elif (score >= signal_thresholds[SignalStatus.WARNING]):
                status   = SignalStatus.WARNING
                severity = 'moderate'
                
            else:
                status   = SignalStatus.PASSED
                severity = 'normal'
            
            # Get explanation from constants
            explanation = METRIC_EXPLANATIONS[metric_type][severity]
            
            # Create signal
            signal      = DetectionSignal(name        = self.detector_registry[metric_type][0],
                                          metric_type = metric_type,
                                          score       = score,
                                          status      = status,
                                          explanation = explanation,
                                         )
            
            signals.append(signal)
        
        # Sort signals by score (highest first)
        signals.sort(key = lambda s: s.score, reverse = True)
        
        return signals


    def _aggregate_scores(self, metric_results: dict) -> float:
        """
        Aggregate individual metric scores using weighted average
        
        Arguments:
        ----------
            metric_results { dict } : Dictionary mapping MetricType to float score
        
        Returns:
        --------
                { float }           : Overall suspicion score [0.0, 1.0]
        """
        total_score  = 0.0
        total_weight = 0.0
        
        for metric_type, result in metric_results.items():
            weight        = self.weights.get(metric_type, 0.0)
            total_score  += result.score * weight
            total_weight += weight
        
        
        # Get Aggregated Score
        if (total_weight > 0):
            # Normalize
            overall_score = total_score / total_weight
        
        else:
            # Neutral if no valid weights
            overall_score = 0.5  
        
        logger.debug(f"Aggregated score: {overall_score:.3f}")
        
        return float(np.clip(overall_score, 0.0, 1.0))


    def _determine_status(self, overall_score: float) -> DetectionStatus:
        """
        Determine binary status from overall score
        
        Arguments:
        ----------
            overall_score { float } : Aggregated suspicion score
        
        Returns:
        --------
            { DetectionStatus }     : LIKELY_AUTHENTIC or REVIEW_REQUIRED
        """
        # Extract review threshold either from threshold_manager or deault to settings value
        review_threshold = (self.threshold_manager.get_review_threshold() if self.threshold_manager else settings.REVIEW_THRESHOLD)

        if (overall_score >= review_threshold):
            return DetectionStatus.REVIEW_REQUIRED
        
        else:
            return DetectionStatus.LIKELY_AUTHENTIC

    
    def _calculate_confidence(self, metric_results: dict[MetricType, MetricResult], overall_score: float) -> int:
        """
        Tier-1 confidence calculator based on:
        - metric agreement
        - metric reliability
        - decision boundary distance
        """
        scores                 = [result.score for result in metric_results.values()]
        score_variance         = np.var(scores)

        # If all metrics failed, confidence must be low
        if all(isinstance(result.details, dict) and "error" in result.details for result in metric_results.values()):
            return int(SIGNAL_CONFIDENCE_PARAMS.MIN_CONFIDENCE * 100)

        # Agreement confidence
        agreement_confidence   = 1.0 - min(score_variance / SIGNAL_CONFIDENCE_PARAMS.VARIANCE_NORM, 1.0)

        # Reliability confidence
        confidences            = [result.confidence for result in metric_results.values() if result.confidence is not None]
        reliability_confidence = float(np.mean(confidences)) if confidences else SIGNAL_CONFIDENCE_PARAMS.DEFAULT_RELIABILITY_CONFIDENCE

        # Distance confidence
        review_threshold       = (self.threshold_manager.get_review_threshold() if self.threshold_manager else settings.REVIEW_THRESHOLD)
        distance_confidence    = min(abs(overall_score - review_threshold) / SIGNAL_CONFIDENCE_PARAMS.DISTANCE_NORM, 1.0)

        logger.debug(f"Confidence breakdown | agreement={agreement_confidence:.2f}, reliability={reliability_confidence:.2f}, distance={distance_confidence:.2f}")

        confidence             = (SIGNAL_CONFIDENCE_PARAMS.AGREEMENT_WEIGHT * agreement_confidence +
                                  SIGNAL_CONFIDENCE_PARAMS.RELIABILITY_WEIGHT * reliability_confidence + 
                                  SIGNAL_CONFIDENCE_PARAMS.DISTANCE_WEIGHT * distance_confidence
                                 )

        return int(np.clip(confidence, 0.0, 1.0) * 100)