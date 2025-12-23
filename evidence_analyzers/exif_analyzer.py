# Dependencies
from PIL import Image
from typing import List
from typing import Dict
from pathlib import Path
from PIL import ExifTags
from typing import Optional
from datetime import datetime
from utils.logger import get_logger
from config.constants import EvidenceType
from config.schemas import EvidenceResult
from config.constants import EvidenceStrength
from config.constants import EvidenceDirection
from config.constants import EXIF_ANALYSIS_PARAMS


# Setup Logging
logger = get_logger(__name__)


class ExifAnalyzer:
    """
    EXIF analyzer produces declarative evidence only: No probabilistic inference
    """
    AI_SOFTWARE_FINGERPRINTS = {"sdxl",
                                "dall-e",
                                "dall·e",
                                "openai",
                                "imagen",
                                "runway",
                                "comfyui",
                                "firefly",
                                "novelai",
                                "craiyon",
                                "leonardo",
                                "midjourney",
                                "adobe sensei",
                                "automatic1111",
                                "waifu diffusion",
                                "stable diffusion",
                                "bing image creator",
                               }

    CAMERA_FIELDS            = {"Make",
                                "Model", 
                                "LensModel",
                               }

    TIME_FIELDS              = {"DateTime", 
                                "DateTimeOriginal", 
                                "DateTimeDigitized",
                               }
    
    AI_INDICATOR_FIELDS      = {"Artist",
                                "Software",
                                "XPComment",
                                "UserComment",
                                "ImageDescription",
                                "ProcessingSoftware",
                               }

    TIMESTAMP_FORMATS        = ["%Y:%m:%d %H:%M:%S",
                                "%Y-%m-%d %H:%M:%S",
                                "%Y:%m:%d %H:%M:%S.%f",
                               ]

    PHOTO_METADATA_FIELDS    = {"ISO",
                                "FNumber",
                                "FocalLength",
                                "ExposureTime",
                                "ISOSpeedRatings",
                               }

    SUSPICIOUS_PATTERNS      = {"unknown",
                                "none",
                                "camera",
                                "test",
                                "generic",
                                "placeholder",
                                "sample",
                               }


    def analyze(self, image_path: Path) -> List[EvidenceResult]:
        evidence = list()
        logger.debug(f"Starting EXIF analysis for {image_path}")

        try:
            image    = Image.open(fp = image_path, mode = "r")
            exif_raw = image.getexif()

            if not exif_raw:
                logger.info("No EXIF metadata found")
                evidence.append(self._missing_exif())
                return evidence

            exif = self._normalize_exif(exif_raw = exif_raw)
            logger.debug(f"Normalized EXIF fields: {list(exif.keys())}")

            evidence.extend(self._analyze_ai_indicators(exif = exif))
            evidence.extend(self._analyze_camera_presence(exif = exif))
            evidence.extend(self._analyze_timestamp_consistency(exif = exif))
            evidence.extend(self._analyze_suspicious_combinations(exif = exif))

        except Exception as e:
            logger.exception("EXIF parsing failed")
            evidence.append(EvidenceResult(source     = EvidenceType.EXIF,
                                           finding    = "EXIF parsing failed",
                                           direction  = EvidenceDirection.INDETERMINATE,
                                           strength   = EvidenceStrength.WEAK,
                                           confidence = 0.0,
                                           details    = {"error": str(e)},
                                           analyzer   = "exif_analyzer",
                                          )
                           )
        
        logger.debug(f"EXIF analysis completed with {len(evidence)} evidence items")
        return evidence

    
    def _normalize_exif(self, exif_raw) -> Dict[str, str]:
        """
        Normalize EXIF tags to human-readable names
        """
        normalized_exif = dict()
        
        for tag, value in exif_raw.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            
            # Convert value to string: handling bytes
            if isinstance(value, bytes):
                try:
                    value_str = value.decode('utf-8', errors = 'ignore')
                
                except:
                    value_str = str(value)
            
            else:
                value_str = str(value)
            
            normalized_exif[tag_name] = value_str
        
        return normalized_exif


    def _missing_exif(self) -> EvidenceResult:
        """
        Missing EXIF is suspicious but not conclusive
        """
        missing_exif = EvidenceResult(source     = EvidenceType.EXIF,
                                      finding    = "No EXIF metadata present (common in AI images and processed web images)",
                                      direction  = EvidenceDirection.INDETERMINATE,
                                      strength   = EvidenceStrength.WEAK,
                                      confidence = EXIF_ANALYSIS_PARAMS.MISSING_EXIF_CONFIDENCE,
                                      details    = {"note": "Missing EXIF alone is not conclusive"},
                                      analyzer   = "exif_analyzer",
                                     )

        return missing_exif


    def _analyze_ai_indicators(self, exif: Dict[str, str]) -> List[EvidenceResult]:
        """
        Check multiple EXIF fields for AI tool indicators
        """
        evidence = list()
        
        for field_name in self.AI_INDICATOR_FIELDS:
            field_value = exif.get(field_name, "").lower()
            
            if not field_value:
                continue
                
            logger.debug(f"Checking {field_name}: {field_value}")
            
            for fingerprint in self.AI_SOFTWARE_FINGERPRINTS:
                if (fingerprint in field_value):
                    logger.warning(f"AI software fingerprint detected in {field_name}: {fingerprint}")
                    evidence.append(EvidenceResult(source     = EvidenceType.EXIF,
                                                   finding    = f"EXIF {field_name} tag matches known AI tool: '{field_value}'",
                                                   direction  = EvidenceDirection.AI_GENERATED,
                                                   strength   = EvidenceStrength.STRONG,
                                                   confidence = EXIF_ANALYSIS_PARAMS.AI_FINGERPRINT_CONFIDENCE,
                                                   details    = {"field"       : field_name, 
                                                                 "value"       : field_value, 
                                                                 "fingerprint" : fingerprint
                                                                },
                                                   analyzer   = "exif_analyzer",
                                                  )
                                   )
                    break

        return evidence


    def _analyze_camera_presence(self, exif: Dict[str, str]) -> List[EvidenceResult]:
        """
        Analyze camera metadata for authenticity indicators
        """
        evidence   = list()
        
        make       = exif.get("Make")
        model      = exif.get("Model")
        lens       = exif.get("LensModel")

        if (make and model):
            logger.debug(f"Camera metadata found: {make} {model}")
            
            confidence = EXIF_ANALYSIS_PARAMS.CAMERA_BASE_CONFIDENCE
            details    = {"make": make, "model": model}
            
            if lens:
                confidence = EXIF_ANALYSIS_PARAMS.CAMERA_WITH_LENS_CONFIDENCE
                details["lens"] = lens
                logger.debug(f"Lens metadata found: {lens}")
            
            if self._is_suspicious_camera_data(make = make, model = model):
                logger.warning(f"Suspicious camera metadata: {make} {model}")
                evidence.append(EvidenceResult(source     = EvidenceType.EXIF,
                                               finding    = f"Suspicious camera metadata detected: {make} {model}",
                                               direction  = EvidenceDirection.INDETERMINATE,
                                               strength   = EvidenceStrength.WEAK,
                                               confidence = EXIF_ANALYSIS_PARAMS.SUSPICIOUS_CAMERA_CONFIDENCE,
                                               details    = details,
                                               analyzer   = "exif_analyzer",
                                              )
                               )
            else:
                evidence.append(EvidenceResult(source     = EvidenceType.EXIF,
                                               finding    = f"Camera metadata present: {make} {model}",
                                               direction  = EvidenceDirection.AUTHENTIC,
                                               strength   = EvidenceStrength.MODERATE,
                                               confidence = confidence,
                                               details    = details,
                                               analyzer   = "exif_analyzer",
                                              )
                               )
        else:
            logger.info("No camera metadata present")
        
        return evidence


    def _is_suspicious_camera_data(self, make: str, model: str) -> bool:
        """
        Check if camera data looks fake or suspicious
        """
        make_lower  = make.lower()
        model_lower = model.lower()
        
        for pattern in self.SUSPICIOUS_PATTERNS:
            if ((pattern in make_lower) or (pattern in model_lower)):
                return True
        
        return False


    def _analyze_timestamp_consistency(self, exif: Dict[str, str]) -> List[EvidenceResult]:
        """
        Check for timestamp inconsistencies
        """
        timestamps = dict()

        for field in self.TIME_FIELDS:
            if (field not in exif):
                continue
                
            parsed_time = self._parse_timestamp(timestamp_str = exif[field])
            
            if parsed_time:
                timestamps[field] = parsed_time

        if (len(timestamps) < 2):
            return []

        time_values   = list(timestamps.values())
        delta         = max(time_values) - min(time_values)
        delta_seconds = delta.total_seconds()
        
        logger.debug(f"Timestamp delta: {delta_seconds} seconds across {len(timestamps)} fields")
        
        if (delta_seconds > EXIF_ANALYSIS_PARAMS.TIMESTAMP_DELTA_THRESHOLD):
            logger.warning(f"Inconsistent EXIF timestamps detected: {delta_seconds}s delta")
            return [EvidenceResult(source     = EvidenceType.EXIF,
                                   finding    = f"Inconsistent EXIF timestamps ({delta_seconds:.1f}s difference)",
                                   direction  = EvidenceDirection.INDETERMINATE,
                                   strength   = EvidenceStrength.WEAK,
                                   confidence = EXIF_ANALYSIS_PARAMS.TIMESTAMP_INCONSISTENCY_CONFIDENCE,
                                   details    = {"delta_seconds" : delta_seconds,
                                                 "timestamps"    : {k: v.isoformat() for k, v in timestamps.items()},
                                                },
                                   analyzer   = "exif_analyzer",
                                  )
                   ]

        return []


    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse timestamp with multiple format attempts
        """
        for fmt in self.TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(timestamp_str, fmt)
            
            except (ValueError, TypeError):
                continue
        
        logger.debug(f"Could not parse timestamp: {timestamp_str}")
        return None


    def _analyze_suspicious_combinations(self, exif: Dict[str, str]) -> List[EvidenceResult]:
        """
        Detect suspicious combinations of EXIF data
        """
        evidence = list()
        
        has_camera         = exif.get("Make") and exif.get("Model")
        has_photo_metadata = any([exif.get(field) for field in self.PHOTO_METADATA_FIELDS])
        
        if (has_camera and not has_photo_metadata):
            logger.warning("Camera metadata present but missing photographic settings")
            evidence.append(EvidenceResult(source     = EvidenceType.EXIF,
                                           finding    = "Camera identified but photographic metadata missing (suspicious)",
                                           direction  = EvidenceDirection.INDETERMINATE,
                                           strength   = EvidenceStrength.WEAK,
                                           confidence = EXIF_ANALYSIS_PARAMS.MISSING_PHOTO_METADATA_CONFIDENCE,
                                           details    = {"has_camera"        : True,
                                                         "missing_settings"  : list(self.PHOTO_METADATA_FIELDS),
                                                        },
                                           analyzer   = "exif_analyzer",
                                          )
                           )
        
        for field in self.TIME_FIELDS:
            if (field not in exif):
                continue
                
            timestamp = self._parse_timestamp(timestamp_str = exif[field])
            
            if (timestamp and self._is_suspicious_timestamp(dt = timestamp)):
                logger.warning(f"Suspicious timestamp detected: {timestamp}")
                evidence.append(EvidenceResult(source     = EvidenceType.EXIF,
                                               finding    = f"Suspicious timestamp pattern in {field}",
                                               direction  = EvidenceDirection.INDETERMINATE,
                                               strength   = EvidenceStrength.WEAK,
                                               confidence = EXIF_ANALYSIS_PARAMS.SUSPICIOUS_TIMESTAMP_CONFIDENCE,
                                               details    = {"field"     : field,
                                                             "timestamp" : timestamp.isoformat(),
                                                             "reason"    : "Suspiciously round time (midnight or all zeros)",
                                                            },
                                               analyzer   = "exif_analyzer",
                                              )
                               )
                break
        
        return evidence


    def _is_suspicious_timestamp(self, dt: datetime) -> bool:
        """
        Check if timestamp looks fake: too perfect/round
        """
        if ((dt.hour == 0) and (dt.minute == 0) and (dt.second == 0)):
            return True
        
        if (dt.year < EXIF_ANALYSIS_PARAMS.MIN_VALID_YEAR):
            return True
        
        if (dt.year > datetime.now().year + EXIF_ANALYSIS_PARAMS.MAX_FUTURE_YEARS):
            return True
        
        return False