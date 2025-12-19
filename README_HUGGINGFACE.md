---
title: AI Image Screener
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - ai-detection
  - image-forensics
  - computer-vision
  - content-moderation
  - screening-tool
---

# AI Image Screener 🔍

**A transparent, unsupervised first-pass screening system for identifying images requiring human review**

## Overview

AI Image Screener is a multi-metric ensemble system that analyzes images using five independent statistical detectors to identify potential AI-generated content. Unlike black-box classifiers, this system provides full explainability with per-metric breakdowns and human-readable explanations.

**Important**: This is a **screening tool, not a verdict engine**. It flags images for human review rather than making definitive "real vs fake" classifications.

## How It Works

The system analyzes five distinct image characteristics:

1. **Gradient-Field PCA (30%)**: Detects lighting inconsistencies typical of diffusion models
2. **Frequency Analysis (25%)**: Identifies unnatural spectral energy distributions via FFT
3. **Noise Pattern Analysis (20%)**: Detects missing or artificial sensor noise
4. **Texture Statistics (15%)**: Identifies overly smooth or repetitive regions
5. **Color Distribution (10%)**: Flags unnatural saturation and color patterns

Each metric produces a score (0.0-1.0), which are combined using weighted ensemble aggregation.

## Expected Performance

**Detection Rates (Honest Estimates):**
- Consumer AI tools (2022-2023): 80-90%
- Stable Diffusion 1.x/2.x: 70-80%
- Midjourney v5/v6: 55-70%
- DALL-E 3 / Gemini Imagen 3: 40-55%
- Post-processed AI images: 30-45%

**False Positive Rate**: ~10-20% on authentic photos (especially HDR, macro, long-exposure)

## Usage

### Web Interface

1. Click "Use this Space" above
2. Upload single or multiple images (max 50 per batch)
3. View results with detailed metric breakdowns
4. Export reports in CSV or PDF format

### API Access

```bash
# Single image analysis
curl -X POST https://huggingface.co/spaces/YOUR_USERNAME/ai-image-screener/api/analyze/image \
  -F "file=@image.jpg"

# Batch analysis
curl -X POST https://huggingface.co/spaces/YOUR_USERNAME/ai-image-screener/api/analyze/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.png"
```

See full API documentation at `/docs` endpoint.

## Limitations

⚠️ **This system has known limitations:**

- **Not adversarially robust**: Simple post-processing can defeat detection
- **Declining effectiveness**: Detection rates decrease as AI models improve
- **False positives**: 10-20% of real photos may be flagged (HDR, macro, heavily edited)
- **No semantic understanding**: Cannot detect deepfakes, inpainting, or prompt-guided generation

## Appropriate Use Cases

✅ **Suitable for:**
- Content moderation pre-screening (reduces human workload)
- Journalism workflows (identifies images needing verification)
- Stock photo platforms (flags for manual review)
- Legal discovery (prioritizes suspicious documents)

❌ **Not suitable for:**
- Law enforcement as sole evidence
- Automated content rejection without human review
- High-stakes decisions (criminal prosecution, copyright disputes)

## Technical Details

- **Framework**: FastAPI (Python 3.11+)
- **Processing Time**: 2-4 seconds per image
- **Dependencies**: OpenCV, NumPy, SciPy, ReportLab
- **No ML Models**: Purely statistical detection (no GPU required)

## Credits

**Author**: Satyaki Mitra (Data Scientist, AI-ML Practitioner)

**Research Foundations**:
- Gragnaniello et al. (2021) - Gradient analysis for GAN detection
- Dzanic et al. (2020) - Fourier spectrum discrepancies
- Kirchner & Johnson (2019) - Sensor pattern noise analysis
- Nataraj et al. (2019) - Co-occurrence matrix detection
- Marra et al. (2019) - GAN-specific artifacts

## License

MIT License - See [LICENSE](LICENSE) for details

## Links

- 📖 [Full Documentation](https://github.com/satyakimitra/ai-image-screener)
- 🏗️ [Architecture Details](https://github.com/satyakimitra/ai-image-screener/blob/main/docs/Architecture.md)
- 📊 [Case Study Analysis](https://github.com/satyakimitra/ai-image-screener/blob/main/docs/CaseStudy_Analysis.md)
- 🔬 [API Reference](https://github.com/satyakimitra/ai-image-screener/blob/main/docs/API.md)

---

**Disclaimer**: Results are indicative and should be verified manually for critical applications. This system provides screening assistance, not definitive judgments.