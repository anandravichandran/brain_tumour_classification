/**
 * ML Service client — forwards images to Python FastAPI and validates responses.
 * Never loads PyTorch/TensorFlow in Node.
 */
import axios, { AxiosError } from 'axios';
import FormData from 'form-data';
import { env } from '../config/env';
import { logger } from '../utils/logger';
import { MLPredictResponseSchema } from '../schemas/analysis.schema';
import { AppError, ErrorCodes } from '../utils/errors';
import type { MLAnalysisResult, MLHealthResponse, MLModelInfoResponse } from '../types/ml.types';

const mlClient = axios.create({
  baseURL: env.mlServiceUrl,
  timeout: 120_000, // 2 min — ML inference can be slow on CPU
});

export async function sendToMLService(
  fileBuffer: Buffer,
  filename: string,
  mimetype: string,
): Promise<MLAnalysisResult> {
  const formData = new FormData();
  formData.append('file', fileBuffer, {
    filename,
    contentType: mimetype,
    knownLength: fileBuffer.length,
  });

  try {
    logger.debug(`Sending ${filename} (${(fileBuffer.length / 1024).toFixed(1)}KB) to ML service`);

    const response = await mlClient.post('/predict', formData, {
      headers: formData.getHeaders(),
    });

    // Validate ML response structure with Zod
    const parsed = MLPredictResponseSchema.safeParse(response.data);
    if (!parsed.success) {
      logger.error(`Invalid ML response schema: ${JSON.stringify(parsed.error.errors)}`);
      throw new AppError(
        ErrorCodes.INTERNAL_SERVER_ERROR,
        'ML service returned an invalid response',
        502,
      );
    }

    logger.info(
      `ML result: ${parsed.data.analysis.prediction} | ` +
      `confidence=${parsed.data.analysis.confidence}`
    );

    return parsed.data.analysis;

  } catch (err) {
    if (err instanceof AppError) throw err;

    if (err instanceof AxiosError) {
      if (err.code === 'ECONNREFUSED' || err.code === 'ENOTFOUND') {
        throw new AppError(
          ErrorCodes.ML_SERVICE_UNAVAILABLE,
          'ML service is not reachable. Please ensure the Python service is running.',
          503,
        );
      }

      if (err.response?.status === 400) {
        const mlError = err.response.data?.error;
        throw new AppError(
          ErrorCodes.INVALID_FILE,
          mlError?.message ?? 'Invalid image file',
          400,
        );
      }

      if (err.response?.status === 500) {
        const mlError = err.response.data?.error;
        throw new AppError(
          ErrorCodes.CLASSIFICATION_FAILED,
          mlError?.message ?? 'ML inference failed',
          500,
        );
      }
    }

    logger.error(`ML service error: ${err instanceof Error ? err.message : String(err)}`);
    throw new AppError(
      ErrorCodes.ML_SERVICE_UNAVAILABLE,
      'Failed to communicate with ML service',
      503,
    );
  }
}

export async function getMLHealth(): Promise<MLHealthResponse> {
  const response = await mlClient.get<MLHealthResponse>('/health');
  return response.data;
}

export async function getMLModelInfo(): Promise<MLModelInfoResponse> {
  const response = await mlClient.get<MLModelInfoResponse>('/model-info');
  return response.data;
}
