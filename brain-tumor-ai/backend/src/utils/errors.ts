/** Standard API error codes shared across backend. */
export const ErrorCodes = {
  INVALID_FILE: 'INVALID_FILE',
  UNSUPPORTED_FILE_TYPE: 'UNSUPPORTED_FILE_TYPE',
  FILE_TOO_LARGE: 'FILE_TOO_LARGE',
  CORRUPTED_IMAGE: 'CORRUPTED_IMAGE',
  MODEL_LOAD_FAILED: 'MODEL_LOAD_FAILED',
  CLASSIFICATION_FAILED: 'CLASSIFICATION_FAILED',
  SEGMENTATION_FAILED: 'SEGMENTATION_FAILED',
  GRADCAM_FAILED: 'GRADCAM_FAILED',
  ML_SERVICE_UNAVAILABLE: 'ML_SERVICE_UNAVAILABLE',
  DEEPSEEK_FAILED: 'DEEPSEEK_FAILED',
  INTERNAL_SERVER_ERROR: 'INTERNAL_SERVER_ERROR',
} as const;

export type ErrorCode = (typeof ErrorCodes)[keyof typeof ErrorCodes];

export class AppError extends Error {
  public readonly code: ErrorCode;
  public readonly statusCode: number;

  constructor(code: ErrorCode, message: string, statusCode = 500) {
    super(message);
    this.name = 'AppError';
    this.code = code;
    this.statusCode = statusCode;
  }
}
