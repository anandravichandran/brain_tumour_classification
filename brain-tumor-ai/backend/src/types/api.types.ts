/** Frontend-facing API response types. */
import type { MLAnalysisResult } from './ml.types';

export interface ApiSuccessResponse<T> {
  success: true;
  data: T;
}

export interface ApiErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
  };
}

export type ApiResponse<T> = ApiSuccessResponse<T> | ApiErrorResponse;

export interface AnalyzeResponse {
  success: true;
  analysis: MLAnalysisResult;
  explanation?: string;
}
