/** TypeScript types for ML service communication. */

export interface MLSegmentationResult {
  detected: boolean;
  tumor_pixels: number | null;
  tumor_percentage: number | null;
  physical_area_mm2: number | null;
  message: string | null;
}

export interface MLVisualizationsResult {
  original: string;
  mask: string | null;
  overlay: string | null;
  gradcam: string | null;
  gradcam_overlay: string | null;
}

export interface MLModelInfo {
  classifier: string;
  segmenter: string;
  version: string;
  classifier_available: boolean;
  segmenter_available: boolean;
}

export interface MLAnalysisResult {
  prediction: 'tumor' | 'no_tumor';
  confidence: number;
  raw_score: number;
  segmentation: MLSegmentationResult;
  visualizations: MLVisualizationsResult;
  model: MLModelInfo;
}

export interface MLPredictResponse {
  success: boolean;
  analysis: MLAnalysisResult;
}

export interface MLHealthResponse {
  status: string;
  classifier_ready: boolean;
  segmenter_ready: boolean;
  tensorflow_version: string;
}

export interface MLModelInfoResponse {
  classifier: Record<string, unknown>;
  segmenter: Record<string, unknown>;
}
