import { Request, Response, NextFunction } from 'express';
import { AppError } from '../utils/errors';
import { logger } from '../utils/logger';

/** Global error handler — never exposes stack traces to clients. */
export function errorMiddleware(
  err: unknown,
  _req: Request,
  res: Response,
  _next: NextFunction
): void {
  if (err instanceof AppError) {
    res.status(err.statusCode).json({
      success: false,
      error: { code: err.code, message: err.message },
    });
    return;
  }

  // Multer errors
  if (err instanceof Error && err.message.includes('file size')) {
    res.status(413).json({
      success: false,
      error: { code: 'FILE_TOO_LARGE', message: err.message },
    });
    return;
  }

  if (err instanceof Error && err.message.includes('Unsupported file type')) {
    res.status(415).json({
      success: false,
      error: { code: 'UNSUPPORTED_FILE_TYPE', message: err.message },
    });
    return;
  }

  logger.error(`Unhandled error: ${err instanceof Error ? err.message : String(err)}`);

  res.status(500).json({
    success: false,
    error: {
      code: 'INTERNAL_SERVER_ERROR',
      message: 'An unexpected error occurred',
    },
  });
}
