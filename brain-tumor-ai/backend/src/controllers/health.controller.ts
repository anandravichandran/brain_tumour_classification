import { Request, Response, NextFunction } from 'express';
import { getMLHealth, getMLModelInfo } from '../services/ml.service';
import { logger } from '../utils/logger';

export async function healthController(_req: Request, res: Response): Promise<void> {
  try {
    const mlHealth = await getMLHealth();
    res.status(200).json({
      status: 'ok',
      service: 'brain-tumor-backend',
      ml_service: mlHealth,
    });
  } catch {
    res.status(200).json({
      status: 'degraded',
      service: 'brain-tumor-backend',
      ml_service: { status: 'unreachable' },
    });
  }
}

export async function modelInfoController(
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const info = await getMLModelInfo();
    res.status(200).json({ success: true, models: info });
  } catch (err) {
    next(err);
  }
}
