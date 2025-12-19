# AI Image Screener  
>*A practical first-pass AI image screening system for modern workflows (2025)*

---

## 1. Overview

**AI Image Screener** is an MVP-grade, **unsupervised image screening system** designed to **identify images that require human review** based on statistical and physical patterns commonly associated with AI-generated imagery.

This system is **not a “perfect AI detector.”**  
It is intentionally built as a **fast, transparent, first-pass screening tool** that helps teams reduce manual review workload by flagging *obviously suspicious* images at scale.

The product is particularly suited for:

- Content moderation pipelines  
- Journalism and media verification  
- Stock image platforms  
- Legal and compliance pre-screening  
- Marketing and brand-protection workflows  

---

## 2. Core Philosophy

### What this product *is*
- A **workflow efficiency tool**
- A **screening system**, not a verdict engine
- A **transparent and explainable detector**
- A **model-agnostic, unsupervised system**

### What this product *is not*
- ❌ A definitive “real vs fake” classifier  
- ❌ A black-box deep learning detector  
- ❌ A system claiming near-perfect accuracy on 2025 AI models  

The system is built on a simple principle:  
**saving human time is more valuable than chasing perfect detection.**

---

## 3. Problem Statement

By 2025, high-quality AI image generators (e.g., DALL·E 3, Gemini Imagen 3, Midjourney v6+) produce images that are often **indistinguishable to humans** and increasingly difficult for single-method detectors.

Most existing tools fail because they:
- Overpromise accuracy
- Provide ambiguous outputs (“uncertain”, “maybe AI”)
- Rely on opaque ML models users do not trust
- Do not integrate into real operational workflows

---

## 4. Product Positioning

### The key insight

Users **do not need certainty** — they need **prioritization**.

Instead of asking:  
> *“Is this image AI or real?”*

The system answers:  
> *“Does this image require human review?”*

---

## 5. Binary UX Model (Critical Design Decision)

The system intentionally provides **only two outcomes**, ensuring every result is actionable.

### 🟢 LIKELY AUTHENTIC
- No significant AI-generation patterns detected
- Passed all screening checks
- **Does not guarantee authenticity**
- No immediate action required

### 🔴 REVIEW REQUIRED
- One or more detection signals triggered
- Patterns consistent with AI generation
- Confidence score provided for prioritization
- **Manual verification recommended**

This avoids the UX failure of ambiguous or “uncertain” results.

---

## 6. Detection Strategy  
### *(Multi-Signal, Unsupervised Ensemble)*

The system runs **multiple independent statistical detectors** on every image.  
Each detector targets a *different failure mode* of AI image generation.

Each metric produces:
- A **normalized anomaly score** in `[0.0 – 1.0]`
- **Rich intermediate details** for explainability and reporting

### Implemented Metrics (`metrics/`)

| Metric | File | Purpose |
|-----|-----|-----|
| Gradient-Field PCA | `metrics/gradient_field_pca.py` | Detects lighting & gradient inconsistencies typical of diffusion |
| Frequency Analysis (FFT) | `metrics/frequency_analyzer.py` | Identifies unnatural spectral energy distributions |
| Noise Pattern Analysis | `metrics/noise_analyzer.py` | Detects missing or artificial sensor noise |
| Texture Statistics | `metrics/texture_analyzer.py` | Identifies overly smooth or uniform regions |
| Color Distribution | `metrics/color_analyzer.py` | Flags unnatural saturation and color histograms |

No single metric is relied upon in isolation.

---

## 7. Score Aggregation & Decision Logic

### Aggregation

All metric outputs are combined using a **weighted ensemble strategy**:

- Implemented in: `metrics/aggregator.py`
- Metric weights are configurable
- No single signal can dominate the final decision
- Robust to individual metric failure

### Thresholding

Final decisions are derived from calibrated thresholds:

- 🟢 **LIKELY_AUTHENTIC** → score below review cutoff  
- 🔴 **REVIEW_REQUIRED** → score above cutoff  

Thresholds and sensitivity modes are managed via:

- `features/threshold_manager.py`
  - Conservative / Balanced / Aggressive modes
  - Runtime threshold tuning
  - A/B calibration support

---

## 8. Explainability & Transparency

Every analysis result includes:

- Which metrics triggered
- Severity level per metric (PASSED / WARNING / FLAGGED)
- Human-readable explanations
- Optional forensic details for advanced users

This avoids black-box behavior and builds user trust.

---

## 9. Reporting & Export Capabilities

The system generates **production-ready reports without recomputation**.

### Reporters (`reporter/`)

| Format | File | Use Case |
|-----|-----|-----|
| CSV | `reporter/csv_reporter.py` | Workflow integration, moderation queues |
| JSON | `reporter/json_reporter.py` | APIs, automation, auditing |
| PDF | `reporter/pdf_reporter.py` | Legal, compliance, documentation |

All reporting is driven by:

- `features/detailed_result_maker.py`  
  (single source of truth for explanations, findings, and summaries)

---

## 10. Technical Architecture

### High-Level Processing Flow

```bash
Upload Image(s)
      ↓
Validation & Preprocessing (utils/)
      ↓
Parallel Metric Execution (metrics/)
      ↓
Score Aggregation (metrics/aggregator.py)
      ↓
Threshold Decision (features/threshold_manager.py)
      ↓
Detailed Result Assembly (features/detailed_result_maker.py)
      ↓
UI / Reports / API Output
```

---

### Backend & Frontend

**Backend**
- FastAPI (Python 3.11+)
- Async batch processing
- Parallel metric execution
- File-based caching (image hash)
- JSON / CSV / PDF outputs
- Clear API contracts (`docs/API.md`)

**Frontend**
- Single-page HTML (inline CSS + JS)
- Batch upload interface
- Live per-metric progress indicators
- Filterable results table
- One-click export actions

---

## 11. Project Structure

```bash
ai_image_screener/
├── app.py
├── config/
│   ├── settings.py
│   ├── constants.py
│   └── schemas.py
├── metrics/
│   ├── gradient_field_pca.py
│   ├── frequency_analyzer.py
│   ├── noise_analyzer.py
│   ├── texture_analyzer.py
│   ├── color_analyzer.py
│   └── aggregator.py
├── features/
│   ├── batch_processor.py
│   ├── detailed_result_maker.py
│   └── threshold_manager.py
├── reporter/
│   ├── csv_reporter.py
│   ├── json_reporter.py
│   └── pdf_reporter.py
├── utils/
│   ├── logger.py
│   ├── image_processor.py
│   ├── validators.py
│   └── helpers.py
├── data/
│   ├── uploads/
│   ├── reports/
│   └── cache/
├── ui/
├── tests/
└── docs/
```

---

## 12. Performance Expectations *(Honest)*

| Image Source | Expected Detection Rate |
|-------------|------------------------|
| Consumer AI tools (older / free) | 80–90% |
| Stable Diffusion (older variants) | 70–80% |
| Midjourney v5 / v6 | 55–70% |
| DALL·E 3 / Gemini Imagen 3 | 40–55% |
| Post-processed AI images | 30–45% |
| False positives on real images | ~10–20% |

These rates are **appropriate for screening**, not final judgment.

---

## 13. Ethical & Legal Positioning

This system:

- Never claims **“real”** or **“fake”**
- Provides **probabilistic screening only**
- Encourages **human verification**
- Documents methodology **transparently**

This makes it suitable for:

- Legal workflows  
- Journalism  
- Enterprise moderation pipelines  

---

## 14. Intended Audience

- Content moderation teams  
- Journalism & media organizations  
- Stock photo platforms  
- Legal & compliance professionals  
- Researchers & educators  

---

## 15. Final Positioning Statement

**AI Image Screener is not an AI detector.**  

> It is a **first-pass screening system designed to save human time**. 
> It flags what needs review — **fast, explainable, and at scale**.