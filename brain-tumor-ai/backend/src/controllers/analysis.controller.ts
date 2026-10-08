import { Request, Response, NextFunction } from 'express';
import { sendToMLService } from '../services/ml.service';
import { logger } from '../utils/logger';
import { AppError, ErrorCodes } from '../utils/errors';

export async function analyzeController(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.file) {
      throw new AppError(ErrorCodes.INVALID_FILE, 'No file uploaded. Please upload an MRI image.', 400);
    }

    const { originalname, mimetype, buffer, size } = req.file;
    logger.info(`Analyze request: ${originalname} | type=${mimetype} | size=${(size / 1024).toFixed(1)}KB`);

    const analysis = await sendToMLService(buffer, originalname, mimetype);

    res.status(200).json({ success: true, analysis });

  } catch (err) {
    next(err);
  }
}
