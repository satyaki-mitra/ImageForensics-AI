# API Documentation

## Base Information

**Base URL**: `http://localhost:8005`  
**API Version**: `2.0.0`  
**Protocol**: HTTP/HTTPS  
**Content Type**: `application/json` (default)  
**Architecture**: Two-tier evidence-first decision system

---

## Table of Contents

1. [Authentication](#authentication)
2. [Health Check](#health-check)
3. [Single Image Analysis](#single-image-analysis)
4. [Batch Image Analysis](#batch-image-analysis)
5. [Batch Progress Tracking](#batch-progress-tracking)
6. [Report Export](#report-export)
7. [Error Handling](#error-handling)
8. [Rate Limits](#rate-limits)
9. [Data Models](#data-models)
10. [Evidence-First Decision Policy](#evidence-first-decision-policy)
11. [Usage Examples](#usage-examples)

---

## Authentication

**Current Version**: No authentication required (intended for internal deployment)

**Future Versions**: API key authentication planned

```bash
# Planned header format
Authorization: Bearer <api_key>
```

---

## Health Check

### `GET /health`

Check if the API server is operational.

**Request**
```bash
curl -X GET http://localhost:8005/health
```

**Response** (`200 OK`)
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

## Single Image Analysis

### `POST /analyze/image`

Analyze a single image for AI-generation indicators.

**Request**

```bash
curl -X POST http://localhost:8005/analyze/image \
  -F "file=@/path/to/image.jpg"
```

**Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file` | File | Yes | Image file (JPG/PNG/WEBP, max 10MB) |

**Response** (`200 OK`)

```json
{
  "success": true,
  "message": "Image analysis completed",
  "data": {
    "filename": "example.jpg",
    "status": "REVIEW_REQUIRED",
    "final_decision": "SUSPICIOUS_AI_LIKELY",
    "decision_explanation": "Strong AI evidence detected in metadata analysis",
    "overall_score": 0.73,
    "confidence": 73,
    "evidence": [
      {
        "source": "exif",
        "finding": "AI software fingerprint detected in EXIF data",
        "direction": "ai_generated",
        "strength": "strong",
        "confidence": 0.92,
        "analyzer": "exif_analyzer",
        "timestamp": "2024-12-19T14:32:15.123456"
      }
    ],
    "signals": [
      {
        "name": "Gradient Field PCA",
        "metric_type": "gradient",
        "score": 0.81,
        "status": "flagged",
        "explanation": "Detected irregular gradient patterns..."
      }
    ],
    "metric_results": {
      "gradient": {
        "metric_type": "gradient",
        "score": 0.81,
        "confidence": 0.87,
        "details": {
          "eigenvalue_ratio": 0.72,
          "gradient_vectors_sampled": 10000
        }
      }
    },
    "processing_time": 2.34,
    "image_size": [1920, 1080],
    "timestamp": "2024-12-19T14:32:15.123456"
  },
  "timestamp": "2024-12-19T14:32:15.123456"
}
```

**Final Decision Values**

- `CONFIRMED_AI_GENERATED`: Conclusive evidence of AI generation

- `SUSPICIOUS_AI_LIKELY`: Strong evidence or high statistical metrics

- `AUTHENTIC_BUT_REVIEW`: Conflicting or weak evidence

- `MOSTLY_AUTHENTIC`: Strong authentic evidence


**Status Values (Tier-1 only)**

- `LIKELY_AUTHENTIC`: Statistical score < 0.65

- `REVIEW_REQUIRED`: Statistical score >= 0.65


**Signal Status Values**

- `passed`: Score < 0.40

- `warning`: Score >= 0.40 and < 0.70

- `flagged`: Score >= 0.70

---

## Batch Image Analysis

### `POST /analyze/batch`

Analyze multiple images in a single request with parallel processing.

**Request**

```bash
curl -X POST http://localhost:8005/analyze/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -F "files=@image3.webp"
```

**Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `files` | File[] | Yes | Multiple image files (max 50 per batch) |

**Response** (`200 OK`)

```json
{
  "success": true,
  "message": "Batch analysis completed",
  "data": {
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "result": {
      "total_images": 3,
      "processed": 3,
      "failed": 0,
      "results": [
        {
          "filename": "image1.jpg",
          "status": "REVIEW_REQUIRED",
          "final_decision": "SUSPICIOUS_AI_LIKELY",
          "decision_explanation": "Strong evidence detected...",
          "overall_score": 0.73,
          "confidence": 73,
          "evidence": [],
          "signals": [],
          "processing_time": 2.1,
          "image_size": [1920, 1080],
          "timestamp": "2024-12-19T14:32:15.123456"
        }
      ],
      "summary": {
        "CONFIRMED_AI_GENERATED": 0,
        "SUSPICIOUS_AI_LIKELY": 1,
        "AUTHENTIC_BUT_REVIEW": 1,
        "MOSTLY_AUTHENTIC": 1,
        "success_rate": 100,
        "processed": 3,
        "failed": 0,
        "avg_score": 0.510,
        "avg_confidence": 51,
        "avg_proc_time": 2.10
      },
      "total_processing_time": 6.3,
      "timestamp": "2024-12-19T14:32:19.345678"
    }
  },
  "timestamp": "2024-12-19T14:32:19.345678"
}
```

**Batch Constraints**
- Maximum images per batch: **50**
- Maximum file size per image: **10 MB**
- Timeout per image: **30 seconds**
- Total batch timeout: **15 minutes**

---

## Batch Progress Tracking

### `GET /batch/{batch_id}/progress`

Track the progress of a batch analysis job.

**Request**

```bash
curl -X GET http://localhost:8005/batch/550e8400-e29b-41d4-a716-446655440000/progress
```

**Response - Processing** (`200 OK`)

```json
{
  "status": "processing",
  "progress": {
    "current": 7,
    "total": 10,
    "filename": "image_007.jpg"
  }
}
```

**Response - Completed** (`200 OK`)

```json
{
  "status": "completed",
  "progress": {
    "current": 10,
    "total": 10,
    "filename": "image_010.jpg"
  },
  "result": {
    "total_images": 10,
    "processed": 10,
    "failed": 0,
    "results": [],
    "summary": {},
    "total_processing_time": 21.4,
    "timestamp": "2024-12-19T14:35:22.123456"
  }
}
```

**Response - Failed** (`200 OK`)

```json
{
  "status": "failed",
  "error": "Processing timeout exceeded"
}
```

**Status Values**
- `processing`: Batch is currently being analyzed
- `completed`: All images processed successfully
- `failed`: Batch processing encountered fatal error
- `interrupted`: Processing was manually stopped

---

## Report Export

### CSV Export

#### `GET /report/csv/{batch_id}` or `POST /report/csv/{batch_id}`

Download detailed batch analysis as CSV file.

**Request**

```bash
curl -X GET http://localhost:8005/report/csv/550e8400-e29b-41d4-a716-446655440000 \
  -o report.csv
```

**Response**

- Content-Type: `text/csv`
- File download with comprehensive analysis data
- Includes: per-image results, metric breakdowns, forensic details

**CSV Structure**

```text
BATCH DECISION STATISTICS
Total Images,10
Processed,10
Failed,0
CONFIRMED_AI_GENERATED,2
SUSPICIOUS_AI_LIKELY,3
AUTHENTIC_BUT_REVIEW,3
MOSTLY_AUTHENTIC,2

ANALYSIS RESULTS
Filename,Final Decision,Decision Confidence (%),Overall Score,Decision Explanation,Processing Time (s)
image1.jpg,SUSPICIOUS_AI_LIKELY,73,0.73,Strong evidence detected...,2.1

IMAGE 1 DETAILED ANALYSIS
FINAL DECISION
Decision,SUSPICIOUS_AI_LIKELY
Confidence,73%
Explanation,Strong evidence detected...

EVIDENCE SUMMARY
Source,Direction,Strength,Confidence,Finding
exif,ai_generated,strong,0.92,AI software fingerprint detected
```

---

## Error Handling

### Error Response Format

All errors return a standardized JSON structure:

```json
{
  "success": false,
  "message": "Error description",
  "error": "Detailed error message",
  "timestamp": "2024-12-19T14:32:15.123456"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `200` | OK | Request successful |
| `400` | Bad Request | Invalid input (file format, size, etc.) |
| `404` | Not Found | Batch ID not found |
| `413` | Payload Too Large | File size exceeds 10MB |
| `422` | Unprocessable Entity | Validation error |
| `499` | Client Closed Request | Processing interrupted |
| `500` | Internal Server Error | Server-side processing error |

### Common Error Scenarios

**File Too Large**
```json
{
  "success": false,
  "message": "Validation error",
  "error": "File size 12582912 bytes exceeds maximum 10485760 bytes",
  "timestamp": "2024-12-19T14:32:15.123456"
}
```

**Unsupported Format**
```json
{
  "success": false,
  "message": "Validation error",
  "error": "File extension .gif not allowed. Allowed: .jpg, .jpeg, .png, .webp",
  "timestamp": "2024-12-19T14:32:15.123456"
}
```

**Batch Not Found**
```json
{
  "success": false,
  "message": "Batch not found",
  "error": null,
  "timestamp": "2024-12-19T14:32:15.123456"
}
```

**Processing Timeout**
```json
{
  "success": false,
  "message": "Processing timeout",
  "error": "Image analysis exceeded 30 second timeout",
  "timestamp": "2024-12-19T14:32:45.123456"
}
```

---

## Rate Limits

**Current Version**: No rate limiting implemented

**Recommended Production Limits**:
- Single image analysis: **60 requests/minute per IP**
- Batch analysis: **10 requests/minute per IP**
- Report downloads: **30 requests/minute per IP**

---

## Data Models

### APIResponse

```typescript
{
  success: boolean,
  message: string,
  data: object | null,
  error: string | null,
  timestamp: string
}
```

### AnalysisResult

```typescript
{
  filename: string,
  status: "LIKELY_AUTHENTIC" | "REVIEW_REQUIRED",
  final_decision: "CONFIRMED_AI_GENERATED" | "SUSPICIOUS_AI_LIKELY" | 
                  "AUTHENTIC_BUT_REVIEW" | "MOSTLY_AUTHENTIC" | null,
  decision_explanation: string | null,
  overall_score: number,      // 0.0 - 1.0
  confidence: number,         // 0 - 100
  signals: DetectionSignal[],
  evidence: EvidenceResult[],
  metric_results: {
    [key: string]: MetricResult
  },
  processing_time: number,    // seconds
  image_size: [number, number],
  timestamp: string
}
```

### EvidenceResult

```typescript
{
  source: "exif" | "watermark",
  finding: string,
  direction: "ai_generated" | "authentic" | "indeterminate",
  strength: "weak" | "moderate" | "strong" | "conclusive",
  confidence: number | null,   // 0.0 - 1.0
  details: object,
  analyzer: string,
  timestamp: string
}
```

### DetectionSignal

```typescript
{
  name: string,
  metric_type: "gradient" | "frequency" | "noise" | "texture" | "color",
  score: number,        // 0.0 - 1.0
  status: "passed" | "warning" | "flagged",
  explanation: string
}
```

### MetricResult

```typescript
{
  metric_type: "gradient" | "frequency" | "noise" | "texture" | "color",
  score: number,        // 0.0 - 1.0
  confidence: number,   // 0.0 - 1.0
  details: object       // Metric-specific forensic data
}
```

### BatchAnalysisResult

```typescript
{
  total_images: number,
  processed: number,
  failed: number,
  results: AnalysisResult[],
  summary: {
    CONFIRMED_AI_GENERATED: number,
    SUSPICIOUS_AI_LIKELY: number,
    AUTHENTIC_BUT_REVIEW: number,
    MOSTLY_AUTHENTIC: number,
    success_rate: number,
    processed: number,
    failed: number,
    avg_score: number,
    avg_confidence: number,
    avg_proc_time: number
  },
  total_processing_time: number,
  timestamp: string
}
```

---

## Evidence-First Decision Policy

### Decision Hierarchy

- Conclusive Evidence → CONFIRMED_AI_GENERATED

- Strong AI Evidence → SUSPICIOUS_AI_LIKELY

- Strong Authentic Evidence → MOSTLY_AUTHENTIC

- Conflicting/Weak Evidence → AUTHENTIC_BUT_REVIEW

- No Evidence → Fallback to statistical metrics


### Evidence Strength Definitions

- CONCLUSIVE: Cryptographic proof, signed metadata

- STRONG: Clear AI fingerprints, consistent patterns

- MODERATE: Suggestive indicators, plausible patterns

- WEAK: Heuristic hints, non-binding observations

---

## Usage Examples

### Python

```python
import requests

# Single image analysis
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8005/analyze/image',
        files={'file': f}
    )
    result = response.json()
    print(f"Final Decision: {result['data']['final_decision']}")

# Batch analysis
files = [
    ('files', open('img1.jpg', 'rb')),
    ('files', open('img2.png', 'rb'))
]
response = requests.post(
    'http://localhost:8005/analyze/batch',
    files=files
)
batch_result = response.json()
batch_id = batch_result['data']['batch_id']

# Check progress
progress = requests.get(f'http://localhost:8005/batch/{batch_id}/progress').json()

# Download CSV report
csv_response = requests.get(f'http://localhost:8005/report/csv/{batch_id}')
with open('report.csv', 'wb') as f:
    f.write(csv_response.content)
```

### JavaScript (Node.js)

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

// Single image analysis
const form = new FormData();
form.append('file', fs.createReadStream('image.jpg'));

axios.post('http://localhost:8005/analyze/image', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log('Final Decision:', response.data.data.final_decision);
  console.log('Evidence found:', response.data.data.evidence.length);
});

// Batch progress tracking
async function trackProgress(batchId) {
  const response = await axios.get(`http://localhost:8005/batch/${batchId}/progress`);
  console.log(`Progress: ${response.data.progress.current}/${response.data.progress.total}`);
  
  if (response.data.status === 'processing') {
    setTimeout(() => trackProgress(batchId), 1000);
  } else if (response.data.status === 'completed') {
    console.log('Batch completed!');
  }
}
```

### cURL

```bash
# Single image
curl -X POST http://localhost:8005/analyze/image \
  -F "file=@image.jpg" \
  | jq '.data.final_decision, .data.evidence[].finding'

# Batch analysis
curl -X POST http://localhost:8005/analyze/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.png" \
  | jq '.data.batch_id'

# Progress tracking
curl -X GET http://localhost:8005/batch/{batch_id}/progress \
  | jq '.status, .progress'

# CSV report
curl -X GET http://localhost:8005/report/csv/{batch_id} -o report.csv
```

---

## Changelog

### Version 1.0.0 (Current)

- Two-tier evidence-first architecture

- Four-class final decisions

- EXIF and watermark evidence analysis

- Decision policy engine

- Enhanced reporting with evidence


### Planned Features

- C2PA provenance analyzer

- API key authentication

- Webhook callbacks

- Custom threshold configuration

- Real-time streaming API

---

*API Documentation Version: 1.0*  
*Last Updated: December 2025*  
*Author: Satyaki Mitra*