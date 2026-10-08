import { Request, Response, NextFunction } from 'express';
import { generateExplanation } from '../services/deepseek.service';
import { ExplanationRequestSchema } from '../schemas/analysis.schema';
import { AppError, ErrorCodes } from '../utils/errors';
import { logger } from '../utils/logger';

export async function explanationController(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const parsed = ExplanationRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      throw new AppError(
        ErrorCodes.INVALID_FILE,
        `Invalid request body: ${parsed.error.errors.map(e => e.message).join(', ')}`,
        400,
      );
    }

    logger.info(`Explanation request: ${parsed.data.prediction} @ ${parsed.data.confidence}`);
    const explanation = await generateExplanation(parsed.data);

    res.status(200).json({ success: true, explanation });

  } catch (err) {
    next(err);
  }
}
