# API Documentation

## POST `/api/analyze`
Accepts a `multipart/form-data` request with an MRI image and returns the complete analysis.

### Request
- `file`: The MRI image file (JPEG or PNG, Max 10MB)

### Response
```json
{
  "success": true,
  "analysis": {
    "prediction": "tumor",
    "confidence": 0.964,
    "raw_score": 0.964,
    "segmentation": {
      "detected": true,
      "tumor_pixels": 12345,
      "tumor_percentage": 8.7,
      "physical_area_mm2": null,
      "message": null
    },
    "visualizations": {
      "original": "<base64_string>",
      "mask": "<base64_string>",
      "overlay": "<base64_string>",
      "gradcam": "<base64_string>",
      "gradcam_overlay": "<base64_string>"
    },
    "model": {
      "classifier": "VGG16",
      "segmenter": "U-Net",
      "version": "1.0",
      "classifier_available": true,
      "segmenter_available": true
    }
  }
}
```

## POST `/api/explanation`
Generates a human-readable text explanation using DeepSeek.

### Request (JSON)
```json
{
  "prediction": "tumor",
  "confidence": 0.964,
  "tumorPixels": 12345,
  "tumorPercentage": 8.7
}
```

### Response
```json
{
  "success": true,
  "explanation": "The trained VGG-16 model..."
}
```

## GET `/health`
Health check and readiness endpoint.
