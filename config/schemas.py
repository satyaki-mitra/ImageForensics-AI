# Dependencies
from typing import List
from typing import Dict
from pydantic import Field
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from config.constants import MetricType
from config.constants import EvidenceType
from config.constants import SignalStatus
from config.constants import FinalDecision
from config.constants import DetectionStatus
from config.constants import EvidenceStrength
from config.constants import EvidenceDirection


# Metric-Level Structures
class MetricResult(BaseModel):
    """
    Raw metric output for explainability and reporting
    """
    metric_type : MetricType
    score       : float           = Field(..., ge = 0.0, le = 1.0)
    confidence  : Optional[float] = Field(None, ge = 0.0, le = 1.0)
    details     : Optional[Dict]  = Field(default_factory = dict)

    model_config                  = {"json_schema_extra" : {"example" : {"metric_type" : "noise",
                                                                         "score"       : 0.72,
                                                                         "confidence"  : 0.81,
                                                                         "details"     : {"patches_total" : 100,
                                                                                          "patches_valid" : 42,
                                                                                          "mean_noise"    : 1.12,
                                                                                          "cv"            : 0.18
                                                                                         }
                                                                        }
                                                           }
                                    }


class DetectionSignal(BaseModel):
    """
    Individual detection signal result
    """
    name        : str          = Field(..., description = "Metric name")
    metric_type : MetricType
    score       : float        = Field(..., ge = 0.0, le = 1.0, description = "Suspicion score (0=natural, 1=suspicious)")
    status      : SignalStatus
    explanation : str          = Field(..., description = "Human-readable explanation")
    
    model_config               = {"json_schema_extra" : {"example" : {"name"        : "Gradient Pattern",
                                                                      "metric_type" : "gradient",
                                                                      "score"       : 0.73,
                                                                      "status"      : "flagged",
                                                                      "explanation" : "Detected irregular gradient patterns typical of diffusion models."
                                                                     }
                                                        } 
                                 }


# # Evidence-Level Structures
class EvidenceResult(BaseModel):
    """
    Declarative evidence extracted from image metadata, watermarking, or cryptographic provenance systems
    """
    source       : EvidenceType            = Field(..., description = "Evidence source type (exif, watermark, c2pa)")
    finding      : str                     = Field(..., description = "Human-readable description of the evidence")
    direction    : EvidenceDirection       = Field(..., description = "What this evidence supports")
    strength     : EvidenceStrength        = Field(..., description = "How strong or reliable this evidence is")
    confidence   : Optional[float]         = Field(None, ge = 0.0, le = 1.0, description = "Confidence in the evidence extraction itself")
    details      : Dict                    = Field(default_factory = dict, description = "Raw extracted fields or technical metadata")
    analyzer     : str                     = Field(..., description = "Analyzer that produced this evidence (exif_analyzer, watermark_analyzer, etc.)")
    timestamp    : datetime                = Field(default_factory = datetime.now)
    model_config                           = {"json_schema_extra": {"example" : {"source"     : "watermark",
                                                                                 "finding"    : "Midjourney v6 watermark detected",
                                                                                 "direction"  : "ai_generated",
                                                                                 "strength"   : "strong",
                                                                                 "confidence" : 0.92,
                                                                                 "details"    : {"watermark_type" : "DWT",
                                                                                                 "vendor"         : "Midjourney",
                                                                                                 "version"        : "v6"
                                                                                                },
                                                                                 "analyzer"   : "watermark_analyzer"
                                                                                }
                                                                   }
                                             }


# Analysis-Level Structures
class AnalysisResult(BaseModel):
    """
    Single image analysis result
    """
    filename             : str
    overall_score        : float                          = Field(..., ge = 0.0, le = 1.0)
    status               : DetectionStatus
    final_decision       : Optional[FinalDecision]        = Field(None, description = "Authoritative decision after evidence-first policy evaluation")
    decision_explanation : Optional[str]                  = Field(None, description = "Human-readable explanation of final decision")
    confidence           : int                            = Field(..., ge = 0, le = 100, description = "Confidence percentage")
    signals              : List[DetectionSignal]
    metric_results       : Dict[MetricType, MetricResult]
    evidence             : List[EvidenceResult]           = Field(default_factory = list, description = "Declarative evidence extracted before decision policy")
    processing_time      : float                          = Field(..., description = "Processing time in seconds")
    timestamp            : datetime                       = Field(default_factory = datetime.now)
    image_size           : tuple[int, int]                = Field(..., description = "Width x Height")
    

    model_config                                          =  {"json_schema_extra" : {"example" : {"filename"        : "photo_001.jpg",
                                                                                                  "overall_score"   : 0.73,
                                                                                                  "status"          : "REVIEW_REQUIRED",
                                                                                                  "confidence"      : 73,
                                                                                                  "signals"         : [],
                                                                                                  "evidence"        : [],
                                                                                                  "processing_time" : 2.34,
                                                                                                  "image_size"      : [1920, 1080]
                                                                                                 }
                                                                                    }
                                                        }


class BatchAnalysisResult(BaseModel):
    """
    Batch analysis result
    """
    total_images          : int
    processed             : int
    failed                : int
    results               : List[AnalysisResult]
    summary               : Dict[str, float]     = Field(default_factory = dict, description = "Summary statistics")
    total_processing_time : float
    timestamp             : datetime             = Field(default_factory = datetime.now)


# API Wrappers
class APIResponse(BaseModel):
    """
    Standard API response wrapper
    """
    success   : bool
    message   : str
    data      : Optional[Dict] = None
    error     : Optional[str]  = None
    timestamp : datetime       = Field(default_factory = datetime.now)


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status    : str
    version   : str
    uptime    : float
    timestamp : datetime = Field(default_factory = datetime.now)
