import { z } from 'zod';

/** Zod schema for validating ML service /predict response. */
export const SegmentationResultSchema = z.object({
  detected: z.boolean(),
  tumor_pixels: z.number().nullable(),
  tumor_percentage: z.number().nullable(),
  physical_area_mm2: z.number().nullable(),
  message: z.string().nullable().optional(),
});

export const VisualizationsResultSchema = z.object({
  original: z.string().min(1),
  mask: z.string().nullable(),
  overlay: z.string().nullable(),
  gradcam: z.string().nullable(),
  gradcam_overlay: z.string().nullable(),
});

export const ModelInfoSchema = z.object({
  classifier: z.string(),
  segmenter: z.string(),
  version: z.string(),
  classifier_available: z.boolean(),
  segmenter_available: z.boolean(),
});

export const AnalysisResultSchema = z.object({
  prediction: z.enum(['tumor', 'no_tumor']),
  confidence: z.number().min(0).max(1),
  raw_score: z.number().min(0).max(1),
  segmentation: SegmentationResultSchema,
  visualizations: VisualizationsResultSchema,
  model: ModelInfoSchema,
});

export const MLPredictResponseSchema = z.object({
  success: z.literal(true),
  analysis: AnalysisResultSchema,
});

/** Schema for the /api/explanation POST body from frontend. */
export const ExplanationRequestSchema = z.object({
  prediction: z.enum(['tumor', 'no_tumor']),
  confidence: z.number().min(0).max(1),
  tumorPixels: z.number().nullable().optional(),
  tumorPercentage: z.number().nullable().optional(),
});

export type ExplanationRequest = z.infer<typeof ExplanationRequestSchema>;
