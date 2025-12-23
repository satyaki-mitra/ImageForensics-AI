# Dependencies
import pywt
import cv2
import numpy as np
from typing import List
from typing import Tuple
from pathlib import Path
from scipy import fftpack
from scipy.stats import entropy
from utils.logger import get_logger
from scipy.signal import correlate2d
from config.schemas import EvidenceResult
from config.constants import EvidenceType
from config.constants import EvidenceStrength
from config.constants import EvidenceDirection
from utils.image_processor import ImageProcessor
from config.constants import WATERMARK_ANALYSIS_PARAMS


# Setup Logging
logger = get_logger(__name__)


class WatermarkAnalyzer:
    """
    Generic watermark detector using signal processing techniques:
    - Detects invisible watermarks through frequency domain analysis
    - wavelet decomposition, and statistical anomalies - vendor agnostic
    """
    def __init__(self):
        self.image_processor = ImageProcessor()


    def analyze(self, image_path: Path) -> List[EvidenceResult]:
        logger.debug(f"Starting watermark analysis for {image_path}")
        
        evidence = list()
        image    = self.image_processor.load_image(image_path)

        evidence.extend(self._detect_wavelet_watermarks(image = image))
        evidence.extend(self._detect_frequency_watermarks(image = image))
        evidence.extend(self._detect_lsb_steganography(image = image))

        logger.debug(f"Watermark analysis completed with {len(evidence)} findings")
        
        if not evidence:
            return []

        return evidence


    def _detect_wavelet_watermarks(self, image: np.ndarray) -> List[EvidenceResult]:
        """
        Detect watermarks embedded in wavelet domain
        - Many invisible watermarks modify high-frequency wavelet coefficients
        - This is a general technique used by multiple AI generators
        """
        logger.debug("Checking for wavelet-domain watermarks")

        try:
            # Convert to grayscale if needed
            if (len(image.shape) == 3):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            else:
                gray = image.copy()

            # Multi-level wavelet decomposition
            coeffs2                = pywt.dwt2(gray, 'haar')
            cA, (cH, cV, cD)       = coeffs2

            # Analyze statistical properties of high-frequency subbands: Watermarks create anomalous energy distributions
            # Calculate sub-band energies
            energy_approx          = np.var(cA)
            energy_h               = np.var(cH)
            energy_v               = np.var(cV)
            energy_d               = np.var(cD)
            
            total_hf_energy        = energy_h + energy_v + energy_d
            total_energy           = energy_approx + total_hf_energy

            if (total_energy == 0):
                return []

            # High-frequency energy ratio
            hf_ratio               = total_hf_energy / total_energy

            # Watermarks increase high-frequency energy beyond natural levels: 
            # - Natural images : ~0.05-0.15
            # - Watermarked    : ~0.20-0.40
            anomalous_energy       = hf_ratio > WATERMARK_ANALYSIS_PARAMS.HF_ENERGY_RATIO_THRESHOLD

            # Check for statistical anomalies in coefficient distribution: watermarks create non-Gaussian distributions
            kurtosis_h             = self._calculate_kurtosis(data = cH)
            kurtosis_v             = self._calculate_kurtosis(data = cV)
            kurtosis_d             = self._calculate_kurtosis(data = cD)
            
            avg_kurtosis           = (kurtosis_h + kurtosis_v + kurtosis_d) / 3
            
            # Natural images: kurtosis ~3-6, Watermarked: often >8
            anomalous_distribution = avg_kurtosis > WATERMARK_ANALYSIS_PARAMS.KURTOSIS_THRESHOLD

            # Check for periodic patterns (grid-based embedding)
            periodicity_score      = self._detect_periodicity(cH, cV, cD)

            # Combined detection
            detected               =  (anomalous_energy and anomalous_distribution) or ((periodicity_score > WATERMARK_ANALYSIS_PARAMS.PERIODICITY_THRESHOLD) and anomalous_energy)

            if detected:
                confidence = self._calculate_confidence([hf_ratio / WATERMARK_ANALYSIS_PARAMS.HF_ENERGY_RATIO_NORM, 
                                                         min(avg_kurtosis / WATERMARK_ANALYSIS_PARAMS.KURTOSIS_NORM_FACTOR, 1.0),
                                                         periodicity_score
                                                       ])
                
                is_strong  = (confidence >= WATERMARK_ANALYSIS_PARAMS.STRONG_CONFIDENCE_THRESHOLD)
                direction  = (EvidenceDirection.AI_GENERATED if is_strong else EvidenceDirection.INDETERMINATE)
                strength   = (EvidenceStrength.STRONG if is_strong else EvidenceStrength.MODERATE)

                logger.warning(f"Heuristic watermark pattern detected in wavelet domain: (confidence: {confidence:.2f})")
                
                return [EvidenceResult(source     = EvidenceType.WATERMARK,
                                       finding    = "Statistical patterns consistent with invisible watermarking or steganographic embedding detected",
                                       direction  = direction,
                                       strength   = strength,
                                       confidence = confidence,
                                       details    = {"method"               : "wavelet_analysis",
                                                     "note"                 : "Heuristic detection; not cryptographic or vendor watermark verification",
                                                     "high_frequency_ratio" : float(hf_ratio),
                                                     "avg_kurtosis"         : float(avg_kurtosis),
                                                     "periodicity_score"    : float(periodicity_score),
                                                     "wavelet_type"         : "haar"
                                                    },
                                        analyzer  = "watermark_analyzer",
                                       )
                       ]

        except Exception as e:
            logger.error(f"Error in wavelet watermark detection: {e}")

        return []


    def _detect_frequency_watermarks(self, image: np.ndarray) -> List[EvidenceResult]:
        """
        Detect watermarks in frequency domain using FFT analysis: Watermarks often add imperceptible patterns in specific frequency bands
        """
        logger.debug("Checking for frequency-domain watermarks")

        try:
            # Convert to grayscale
            if (len(image.shape) == 3):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            else:
                gray = image.copy()

            # 2D FFT
            fft                = fftpack.fft2(gray)
            fft_shift          = fftpack.fftshift(fft)
            magnitude          = np.abs(fft_shift)

            # Log scale for better visualization of weak signals
            magnitude_log      = np.log1p(magnitude)
            
            # Analyze frequency spectrum
            h, w               = magnitude_log.shape
            center_y, center_x = h // 2, w // 2

            # Check mid to high frequency bands (common watermark location): Divide spectrum into radial bands
            band_anomalies     = list()
            frequency_bands    = [(0.2, 0.4),  # Mid-low frequencies
                                  (0.4, 0.6),  # Mid frequencies
                                  (0.6, 0.8),  # Mid-high frequencies
                                 ]

            for low, high in frequency_bands:
                mask        = self._create_radial_mask(magnitude_log.shape, low, high)
                band_values = magnitude_log[mask]
                
                if (band_values.size == 0):
                    continue

                # Statistical analysis of band
                band_mean   = np.mean(band_values)
                band_std    = np.std(band_values)
                
                # Detect anomalous peaks (watermark signatures)
                threshold   = band_mean + WATERMARK_ANALYSIS_PARAMS.PEAK_STD_MULTIPLIER * band_std
                peaks       = np.sum(band_values > threshold)

                peak_ratio  = peaks / band_values.size
                
                if (peak_ratio > WATERMARK_ANALYSIS_PARAMS.PEAK_RATIO_THRESHOLD):  
                    # More than 5% anomalous values
                    band_anomalies.append({'band'       : (low, high),
                                           'peak_ratio' : float(peak_ratio),
                                           'peak_count' : int(peaks)
                                         })

            # Check for symmetric patterns (common in structured watermarks)
            symmetry_score = self._check_spectral_symmetry(magnitude = magnitude_log)

            detected       = ((len(band_anomalies) >= WATERMARK_ANALYSIS_PARAMS.MIN_ANOMALOUS_BANDS) and 
                              (symmetry_score >  WATERMARK_ANALYSIS_PARAMS.SPECTRAL_SYMMETRY_THRESHOLD))
            
            if detected:
                confidence = self._calculate_confidence([min(len(band_anomalies) / 3, 1.0),
                                                         symmetry_score
                                                       ])
                
                is_strong  = (confidence >= WATERMARK_ANALYSIS_PARAMS.STRONG_CONFIDENCE_THRESHOLD)
                direction  = (EvidenceDirection.AI_GENERATED if is_strong else EvidenceDirection.INDETERMINATE)
                strength   = (EvidenceStrength.STRONG if is_strong else EvidenceStrength.MODERATE)

                logger.warning(f"Heuristic watermark pattern detected in Frequency-domain: (confidence: {confidence:.2f})")
                
                return [EvidenceResult(source     = EvidenceType.WATERMARK,
                                       finding    = "Statistical patterns consistent with invisible watermarking or steganographic embedding detected",
                                       direction  = direction,
                                       strength   = strength,
                                       confidence = confidence,
                                       details    = {"method"          : "frequency_analysis",
                                                     "note"            : "Heuristic detection; not cryptographic or vendor watermark verification",
                                                     "anomalous_bands" : len(band_anomalies),
                                                     "band_details"    : band_anomalies,
                                                     "symmetry_score"  : float(symmetry_score),
                                                    },
                                       analyzer   = "watermark_analyzer",
                                      )
                       ]

        except Exception as e:
            logger.error(f"Error in frequency watermark detection: {e}")

        return []


    def _detect_lsb_steganography(self, image: np.ndarray) -> List[EvidenceResult]:
        """
        Detect steganographic watermarks using LSB (Least Significant Bit) analysis.
        Many watermarking schemes embed data in the LSB planes.
        """
        logger.debug("Checking for LSB steganography")

        try:
            # Analyze all color channels
            if (len(image.shape) == 3):
                channels = cv2.split(image)
            
            else:
                channels = [image]

            channel_results = list()
            
            for idx, channel in enumerate(channels):
                # Extract bit planes
                lsb_plane   = channel & 1         # LSB
                msb_plane   = (channel >> 7) & 1  # MSB for comparison

                # Calculate entropy
                lsb_entropy = self._shannon_entropy(lsb_plane)
                msb_entropy = self._shannon_entropy(msb_plane)

                # Chi-square test for randomness
                chi_square  = self._chi_square_test(lsb_plane)

                # Run test for detecting non-random patterns
                runs        = self._runs_test(lsb_plane)

                channel_results.append({'channel'     : idx,
                                        'lsb_entropy' : float(lsb_entropy),
                                        'msb_entropy' : float(msb_entropy),
                                        'chi_square'  : float(chi_square),
                                        'runs_score'  : float(runs)
                                      })

            # Average results across channels
            avg_lsb_entropy    = np.mean([r['lsb_entropy'] for r in channel_results])
            avg_chi_square     = np.mean([r['chi_square'] for r in channel_results])
            avg_runs           = np.mean([r['runs_score'] for r in channel_results])

            # Detection criteria:
            # - High LSB entropy (>0.72) indicates embedded data
            # - High chi-square indicates non-uniform distribution
            # - Runs test indicates structured patterns
            
            suspicious_entropy = (avg_lsb_entropy > WATERMARK_ANALYSIS_PARAMS.LSB_ENTROPY_THRESHOLD)
            suspicious_chi     = (avg_chi_square > WATERMARK_ANALYSIS_PARAMS.CHI_SQUARE_THRESHOLD)
            suspicious_runs    = (avg_runs > WATERMARK_ANALYSIS_PARAMS.RUNS_SCORE_THRESHOLD)

            detected           = (suspicious_entropy and (suspicious_chi or suspicious_runs))
            
            if detected:
                # Determine strength based on confidence
                confidence = self._calculate_confidence([min((avg_lsb_entropy - WATERMARK_ANALYSIS_PARAMS.LSB_ENTROPY_NORM_BASE) / WATERMARK_ANALYSIS_PARAMS.LSB_ENTROPY_NORM_RANGE, 1.0),
                                                         min(avg_chi_square / WATERMARK_ANALYSIS_PARAMS.CHI_SQUARE_NORM_FACTOR, 1.0),
                                                         avg_runs
                                                       ])
                is_strong  = (confidence >= WATERMARK_ANALYSIS_PARAMS.STRONG_CONFIDENCE_THRESHOLD)
                direction  = (EvidenceDirection.AI_GENERATED if is_strong else EvidenceDirection.INDETERMINATE)
                strength   = (EvidenceStrength.STRONG if is_strong else EvidenceStrength.MODERATE)

                logger.warning(f"Heuristic watermark pattern detected in LSB steganography-domain: (confidence: {confidence:.2f})")
                
                return [EvidenceResult(source     = EvidenceType.WATERMARK,
                                       finding    = "Statistical patterns consistent with invisible watermarking or steganographic embedding detected",
                                       direction  = direction,
                                       strength   = strength,
                                       confidence = confidence,
                                       details    = {"method"          : "lsb_analysis",
                                                     "note"            : "Heuristic detection; not cryptographic or vendor watermark verification",
                                                     "avg_lsb_entropy" : float(avg_lsb_entropy),
                                                     "avg_chi_square"  : float(avg_chi_square),
                                                     "avg_runs_score"  : float(avg_runs),
                                                     "avg_msb_entropy" : float(np.mean([r["msb_entropy"] for r in channel_results])),
                                                     "channel_results" : channel_results
                                                    },
                                       analyzer   = "watermark_analyzer",
                                      )
                       ]

        except Exception as e:
            logger.error(f"Error in LSB steganography detection: {e}")

        return []


    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """
        Calculate kurtosis: measure of distribution tailedness
        """
        data_flat = data.flatten()
        mean      = np.mean(data_flat)
        std       = np.std(data_flat)
        
        if (std == 0):
            return 0.0

        normalized = (data_flat - mean) / std

        return float(np.mean(normalized ** 4))


    def _detect_periodicity(self, *coeffs) -> float:
        """
        Detect periodic patterns in coefficients: grid-based watermarks
        """
        try:
            scores = list()

            for coeff in coeffs:
                # Apply autocorrelation
                autocorr         = correlate2d(coeff, coeff, mode = 'same')
                
                max_val          = np.max(autocorr)
                if (max_val == 0):
                    continue

                autocorr         = autocorr / max_val
                
                # Look for secondary peaks (indicating periodicity)
                center           = tuple(s // 2 for s in autocorr.shape)

                # Remove center peak
                autocorr[center] = 0  
                
                max_secondary    = np.max(autocorr)
                scores.append(max_secondary)
            
            return float(np.mean(scores))

        except:
            return 0.0


    def _create_radial_mask(self, shape: Tuple[int, int], inner_ratio: float, outer_ratio: float) -> np.ndarray:
        """
        Create radial mask for frequency analysis
        """
        h, w               = shape
        center_y, center_x = h // 2, w // 2
        max_radius         = min(center_y, center_x)
        
        y, x               = np.ogrid[:h, :w]
        distances          = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        mask               = (distances >= inner_ratio * max_radius) & (distances < outer_ratio * max_radius)
        
        return mask


    def _check_spectral_symmetry(self, magnitude: np.ndarray) -> float:
        """
        Check for symmetric patterns in frequency spectrum
        """
        try:
            h, w        = magnitude.shape
            left_half   = magnitude[:, :w//2]
            right_half  = np.fliplr(magnitude[:, w//2:])
            
            # Ensure same size
            min_width   = min(left_half.shape[1], right_half.shape[1])
            left_half   = left_half[:, :min_width]
            right_half  = right_half[:, :min_width]
            
            # Calculate correlation
            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            
            return float(abs(correlation)) if not np.isnan(correlation) else 0.0
        
        except:
            return 0.0


    def _shannon_entropy(self, data: np.ndarray) -> float:
        """
        Calculate Shannon entropy
        """
        values, counts = np.unique(data.flatten(), return_counts = True)
        probabilities  = counts / counts.sum()

        return float(entropy(probabilities, base=2))


    def _chi_square_test(self, data: np.ndarray) -> float:
        """
        Chi-square test for uniformity
        """
        values, counts = np.unique(data.flatten(), return_counts = True)
        expected       = len(data.flatten()) / len(values)
        chi_square     = np.sum((counts - expected) ** 2 / expected)

        return float(chi_square)


    def _runs_test(self, data: np.ndarray) -> float:
        """
        Runs test for randomness: normalized score
        """
        flat          = data.flatten()
        median        = np.median(flat)
        runs          = np.sum(np.abs(np.diff((flat > median).astype(int))))
        expected_runs = len(flat) / 2

        if (expected_runs == 0):
            return 0.0

        return float(min(runs / expected_runs, 1.0))


    def _calculate_confidence(self, scores: List[float]) -> float:
        """
        Calculate overall confidence from multiple scores
        """
        valid_scores = [score for score in scores if ((isinstance(score, (int, float))) and (not np.isnan(score)))]
        
        if not valid_scores:
            return 0.0
        
        confidence = np.mean(valid_scores)
        
        # Cap at 0.95
        return float(min(max(confidence, 0.0), WATERMARK_ANALYSIS_PARAMS.CONFIDENCE_CAP)) 