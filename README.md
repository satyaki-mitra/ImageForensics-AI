---
title: ImageForensics-AI
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

<p align="center"> 

# ImageForensics-AI

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=%white)](https://scipy.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-%23E92063.svg?logo=pydantic&logoColor=white)](https://pydantic-docs.helpmanual.io/)
[![Uvicorn](https://img.shields.io/badge/uvicorn-%23009539.svg?logo=uvicorn&logoColor=white)](https://www.uvicorn.org/)


> **A transparent, multi-tier evidence-first screening system for identifying images requiring human review in production workflows**

</p>

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Performance](#-performance--accuracy)
- [License](#-license)

---

## 🎯 Overview

**ImageForensics-AI** is a **tiered evidence-first screening tool** designed to reduce manual review workload by analyzing images through statistical metrics (Tier-1) and declarative evidence (Tier-2), then resolving through a deterministic decision policy.

### What This Is
✅ A workflow efficiency tool with evidence-first architecture  
✅ A transparent, explainable multi-tier detector  
✅ A model-agnostic screening system with declarative evidence  
✅ A first-pass filter with policy-based decisions

### What This Is Not
❌ A definitive "real vs fake" classifier  
❌ A black-box deep learning detector  
❌ A system claiming near-perfect accuracy on 2025 AI models

---

## 🚀 Key Features

- **Tiered Analysis**: 
  - **Tier-1**: 5 statistical metric detectors
  - **Tier-2**: Declarative evidence analyzers (EXIF, watermark)
  - **Decision Layer**: Evidence-first policy engine
- **Multi-Class Decisions**: Four nuanced outcomes beyond binary classification
- **Full Explainability**: Per-metric scores, evidence findings, and human-readable explanations
- **Batch Processing**: Parallel analysis of up to 50 images with progress tracking
- **Multiple Export Formats**: CSV, JSON reports for integration into existing workflows
- **No External Dependencies**: No ML models, no cloud APIs - fully self-contained
- **Production Ready**: FastAPI backend, comprehensive error handling, configurable thresholds

---

## 🏗️ Architecture Overview

### Two-Tier Analysis with Evidence-First Decision

```mermaid
flowchart TD
    %% Overall System Flow
    Input[📁 Input: Image File]
    
    %% Tier 1: Statistical Metrics
    subgraph Tier1 [Tier 1: Statistical Metrics]
        Gradient[Gradient PCA<br/>Eigenvalue Analysis]
        Frequency[Frequency FFT<br/>Spectral Analysis]
        Noise[Noise Pattern<br/>Sensor Noise Analysis]
        Texture[Texture Stats<br/>Smoothness Detection]
        Color[Color Distribution<br/>Saturation & Hue Analysis]
    end
    
    %% Tier 1 Aggregation
    Aggregator1[Signal Aggregator<br/>Weighted Ensemble]
    Status1{Detection Status}
    
    %% Tier 2: Declarative Evidence
    subgraph Tier2 [Tier 2: Declarative Evidence]
        EXIF[EXIF Analyzer<br/>Metadata Forensics]
        Watermark[Watermark Detector<br/>Statistical Patterns]
        Future[C2PA Analyzer<br/>Provenance - Future]
    end
    
    %% Tier 2 Aggregation
    Aggregator2[Evidence Aggregator<br/>Deduplication & Ranking]
    Evidence[(Evidence Results<br/>with Strength & Direction)]
    
    %% Decision Engine
    subgraph Decision [Decision Policy Engine]
        Rule1[Rule 1: Conclusive evidence<br/>overrides all]
        Rule2[Rule 2: Strong evidence ><br/>statistical metrics]
        Rule3[Rule 3: Conflicting evidence → Review Required]
        Rule4[Rule 4: No evidence →<br/>fallback to Tier 1]
    end
    
    %% Final Decisions
    Final1[✅ CONFIRMED_AI_GENERATED<br/>Conclusive evidence]
    Final2[⚠️ SUSPICIOUS_AI_LIKELY<br/>Strong evidence/metrics]
    Final3[🔄 AUTHENTIC_BUT_REVIEW<br/>Conflicting/weak evidence]
    Final4[✓ MOSTLY_AUTHENTIC<br/>Strong authentic evidence]
    
    %% Connections
    Input --> Tier1
    Input --> Tier2
    
    Gradient --> Aggregator1
    Frequency --> Aggregator1
    Noise --> Aggregator1
    Texture --> Aggregator1
    Color --> Aggregator1
    Aggregator1 --> Status1
    
    EXIF --> Aggregator2
    Watermark --> Aggregator2
    Future -.-> Aggregator2
    
    Aggregator2 --> Evidence
    
    Status1 --> Decision
    Evidence --> Decision
    
    Decision --> Rule1
    Decision --> Rule2
    Decision --> Rule3
    Decision --> Rule4
    
    Rule1 --> Final1
    Rule2 --> Final2
    Rule3 --> Final3
    Rule4 --> Final4
    
    %% Styling
    classDef tier1 fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef tier2 fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    classDef final fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class Tier1 tier1
    class Tier2 tier2
    class Decision decision
    class Final1,Final2,Final3,Final4 final
```

### Component Diagram

```mermaid
flowchart TD
    %% Overall System Flow
    Input[📁 Input: Image File<br/>Validated by ImageValidator]
    
    %% Tier 1: Statistical Metrics (Actual Components)
    subgraph T1 [Tier 1: Statistical Metrics (metrics/)]
        G[metrics/gradient_field_pca.py]
        F[metrics/frequency_analyzer.py]
        N[metrics/noise_analyzer.py]
        T[metrics/texture_analyzer.py]
        C[metrics/color_analyzer.py]
    end
    
    %% Tier 1 Aggregation
    SA[metrics/signal_aggregator.py<br/>Signal Aggregator]
    Status{LIKELY_AUTHENTIC<br/>or REVIEW_REQUIRED}
    
    %% Tier 2: Declarative Evidence (Actual Components)
    subgraph T2 [Tier 2: Declarative Evidence (evidence_analyzers/)]
        EX[evidence_analyzers/exif_analyzer.py]
        WM[evidence_analyzers/watermark_analyzer.py]
        EA[evidence_analyzers/evidence_aggregator.py]
    end
    
    Evidence[(Evidence Results:<br/>source, direction, strength, confidence)]
    
    %% Decision Engine
    DP[decision_builders/decision_policy.py<br/>Evidence-First Decision Policy]
    
    %% Final Decisions
    FD1[🔴 CONFIRMED_AI_GENERATED<br/>Conclusive evidence<br/>e.g., cryptographic watermark]
    FD2[🟠 SUSPICIOUS_AI_LIKELY<br/>Strong evidence or high metrics]
    FD3[🟡 AUTHENTIC_BUT_REVIEW<br/>Conflicting/indeterminate evidence]
    FD4[🟢 MOSTLY_AUTHENTIC<br/>Strong authentic evidence]
    
    %% Connections
    Input --> T1
    Input --> T2
    
    G --> SA
    F --> SA
    N --> SA
    T --> SA
    C --> SA
    SA --> Status
    
    EX --> EA
    WM --> EA
    EA --> Evidence
    
    Status --> DP
    Evidence --> DP
    
    DP --> FD1
    DP --> FD2
    DP --> FD3
    DP --> FD4
    
    %% Styling
    classDef metrics fill:#e1f5fe,stroke:#0277bd
    classDef evidence fill:#f3e5f5,stroke:#7b1fa2
    classDef decision fill:#fff8e1,stroke:#ff8f00
    classDef final fill:#e8f5e8,stroke:#43a047
    
    class T1 metrics
    class T2 evidence
    class DP decision
    class FD1,FD2,FD3,FD4 final
```

---

## 🔬 Detection Approach

### Core Philosophy: Evidence-First Decision Making

Instead of probabilistic inference, we apply deterministic rules:
1. **Conclusive Evidence** (e.g., cryptographic watermarks) → Immediate decision
2. **Strong Evidence** (e.g., AI tool fingerprints) → Overrides statistical metrics
3. **Moderate Evidence** → Combined with statistical assessment
4. **No Evidence** → Fallback to statistical metrics

### Tier-1: Statistical Metrics (Unsupervised)

Five orthogonal metrics targeting different AI generation failure modes:

#### 1. **Gradient-Field PCA** (30% weight)
- **Target**: Lighting inconsistencies in diffusion models
- **Method**: Sobel gradients → PCA → eigenvalue ratio analysis
- **Threshold**: Eigenvalue ratio < 0.85 indicates suspicious structure
- **Research Basis**: Gragnaniello et al. 2021

#### 2. **Frequency Analysis (FFT)** (25% weight)
- **Target**: Unnatural spectral energy distributions
- **Method**: 2D FFT → radial spectrum → HF ratio + roughness + deviation
- **Thresholds**: HF ratio outside [0.08, 0.35] indicates anomalies
- **Research Basis**: Dzanic et al. 2020

#### 3. **Noise Pattern Analysis** (20% weight)
- **Target**: Missing or artificial sensor noise
- **Method**: Patch-based Laplacian filtering → MAD estimation → CV + IQR analysis
- **Thresholds**: CV < 0.15 (too uniform) or > 1.2 (too variable)
- **Research Basis**: Kirchner & Johnson 2019

#### 4. **Texture Statistics** (15% weight)
- **Target**: Overly smooth or repetitive regions
- **Method**: Patch-based entropy, contrast, edge density → distribution analysis
- **Thresholds**: >40% smooth patches (smoothness > 0.5) indicates anomalies
- **Research Basis**: Nataraj et al. 2019

#### 5. **Color Distribution** (10% weight)
- **Target**: Impossible or unlikely color patterns
- **Method**: RGB→HSV → saturation analysis + histogram roughness + hue concentration
- **Thresholds**: Mean saturation > 0.65 or top-3 hue bins > 60%
- **Research Basis**: Marra et al. 2019

### Tier-2: Declarative Evidence (Non-Scoring)

#### **EXIF Analyzer**
- **Purpose**: Extract and analyze metadata for authenticity indicators
- **Checks**: AI software fingerprints, camera metadata consistency, timestamp plausibility
- **Output**: Directional evidence (AI_GENERATED/AUTHENTIC/INDETERMINATE) with confidence

#### **Watermark Analyzer**
- **Purpose**: Detect statistical patterns of invisible watermarks
- **Methods**: Wavelet decomposition, frequency domain analysis, LSB steganography detection
- **Output**: Evidence of potential watermarking with strength assessment

---

## 📊 Decision Policy Rules

### Authority Hierarchy
1. **Conclusive Evidence** (EvidenceStrength.CONCLUSIVE) → Immediate override
2. **Strong Evidence** (EvidenceStrength.STRONG) → Overrides contradictory metrics
3. **Moderate Evidence** → Combined with metric confidence
4. **Weak/Indeterminate Evidence** → Supplementary only

### Conflict Resolution
- **AI vs Authentic Evidence**: Strongest evidence type wins; ties → "REVIEW_REQUIRED"
- **Multiple Indeterminate Findings**: ≥2 indeterminate items → "AUTHENTIC_BUT_REVIEW"
- **No Evidence**: Fallback to Tier-1 metric status

### Final Decision Mapping
```python
# Evidence rules take precedence
if conclusive_ai_evidence:
    return FinalDecision.CONFIRMED_AI_GENERATED
elif strong_ai_evidence:
    return FinalDecision.SUSPICIOUS_AI_LIKELY
elif strong_authentic_evidence:
    return FinalDecision.MOSTLY_AUTHENTIC
elif conflicting_moderate_evidence:
    return FinalDecision.AUTHENTIC_BUT_REVIEW

# Fallback to Tier-1 metrics
elif tier1_status == "REVIEW_REQUIRED":
    return FinalDecision.SUSPICIOUS_AI_LIKELY
else:
    return FinalDecision.MOSTLY_AUTHENTIC
```

---

## 🏗️ System Architecture

### Component Structure

```bash
ImageForensics-AI/
├── app.py                              # FastAPI application
├── config/
│   ├── settings.py                     # Environment config
│   ├── constants.py                    # Enums, parameters, explanations
│   └── schemas.py                      # Pydantic models
├── metrics/                            # TIER 1: Statistical detectors
│   ├── gradient_field_pca.py
│   ├── frequency_analyzer.py
│   ├── noise_analyzer.py
│   ├── texture_analyzer.py
│   ├── color_analyzer.py
│   └── signal_aggregator.py            # Metric orchestration
├── evidence_analyzers/                 # TIER 2: Declarative evidence
│   ├── exif_analyzer.py
│   ├── watermark_analyzer.py
│   └── evidence_aggregator.py          # Evidence orchestration
├── decision_builders/
│   └── decision_policy.py              # Evidence-first decision rules
├── features/
│   ├── batch_processor.py              # Batch orchestration
│   ├── threshold_manager.py            # Runtime configuration
│   └── detailed_result_maker.py        # Explainability extraction
├── reporter/
│   ├── csv_reporter.py                 # CSV export
│   └── json_reporter.py                # JSON API responses
├── utils/
│   ├── logger.py                       # Structured logging
│   ├── image_processor.py              # Image utilities
│   ├── validators.py                   # File validation
│   └── helpers.py                      # General utilities
└── ui/
    └── index.html                      # Web interface
```

### Data Flow

```
Image → Validation → [Tier-1 Metrics + Tier-2 Evidence] → Aggregation → Decision Policy → Final Decision → Reporting
                      ↗               ↗                                  ↖
              SignalAggregator  EvidenceAggregator                  DecisionPolicy
```

---

## 📦 Installation

### Prerequisites

- Python 3.11+

- pip

### Setup

```bash
# Clone repository
git clone https://github.com/satyaki-mitra/ImageForensics-AI.git
cd ImageForensics-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data/{uploads,reports,cache} logs

# Run server
python app.py
```

- Server starts at `http://localhost:8005`

---

## 🚀 Quick Start

### Web Interface

- Open `http://localhost:8005` in browser

- Upload images (single or batch)

- View results with evidence breakdown

- Export reports (CSV/JSON)


### API Usage

```bash
# Single image analysis
curl -X POST http://localhost:8005/analyze/image \
  -F "file=@example.jpg"

# Batch analysis
curl -X POST http://localhost:8005/analyze/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.png"

# Get batch progress
curl http://localhost:8005/batch/{batch_id}/progress

# Download CSV report
curl http://localhost:8005/report/csv/{batch_id} -o report.csv
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Server
HOST=localhost
PORT=8005
DEBUG=False

# Detection Thresholds
REVIEW_THRESHOLD=0.65

# Tier-1 Metric Weights (must sum to 1.0)
GRADIENT_WEIGHT=0.30
FREQUENCY_WEIGHT=0.25
NOISE_WEIGHT=0.20
TEXTURE_WEIGHT=0.15
COLOR_WEIGHT=0.10

# Processing
MAX_FILE_SIZE_MB=10
MAX_BATCH_SIZE=50
PARALLEL_PROCESSING=True
MAX_WORKERS=4
```

### Runtime Configuration via API

```python
# Adjust thresholds dynamically
threshold_manager.set_review_threshold(0.70)
threshold_manager.set_metric_weight(MetricType.GRADIENT, 0.35)

# Get recommendations
recommendations = threshold_manager.get_recommendations(score=0.85)
```

---

## 📊 Performance & Accuracy

### Expected Detection Rates

| Scenario | Detection Rate | Notes |
|----------|----------------|-------|
| Consumer AI (2022-2023) | 80–90% | Early generators leave clear traces |
| Modern Diffusion (2024) | 55–70% | Improved physical plausibility |
| Post-processed AI | 30–45% | Compression/filters remove artifacts |
| False Positives (Real) | 10–20% | HDR, macro, long-exposure trigger |


### Processing Performance

- Single Image: 2–4 seconds (metrics + evidence)

- Batch (10): 15–25 seconds (parallel)

- Memory: 50–150 MB per image

- Concurrency: 4 workers (configurable)

---

## ⚠️ Ethical Considerations & Limitations

### Appropriate Use Cases

✅ **Content moderation pre-screening**: Reduces human workload
✅ **Journalism verification workflows**: Identifies images needing source checking
✅ **Stock media platforms**: Flags for manual review before publication
✅ **Academic research**: Transparent methodology for study


### Not Suitable For

❌ **Law enforcement as sole evidence**: Always requires human verification
❌ **Automated content rejection without review**: Human-in-loop required
❌ **High-stakes individual decisions**: e.g., employment, legal proceedings


### Known Limitations

- Evolving Generators: Detection rates decline as AI models improve

- Adversarial Evasion: Intentional post-processing can defeat statistical detectors

- Cultural Bias: Training data biases may affect certain image types

- No Ground Truth: Statistical anomalies don't guarantee AI origin

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👤 Author

Satyaki Mitra
Data Scientist | AI-ML Practitioner

---

## 🙏 Acknowledgments

- Research papers cited for theoretical foundations

- FastAPI team for excellent web framework

- OpenCV and SciPy communities for image processing tools

---

## 🔗 Resources


- Documentation: docs/ directory

---

<p align="center"> 
<i>Built with transparency and evidence-first principles.</i>
<br>
<i>Screening, not certainty. Evidence, not guesswork.</i> 
</p> 





