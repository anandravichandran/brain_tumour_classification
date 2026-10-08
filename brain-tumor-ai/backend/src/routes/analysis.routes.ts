import { Router } from 'express';
import { uploadMiddleware } from '../middleware/upload.middleware';
import { analyzeController } from '../controllers/analysis.controller';

const router = Router();

router.post('/analyze', (req, res, next) => {
  uploadMiddleware(req, res, (err) => {
    if (err) return next(err);
    return analyzeController(req, res, next);
  });
});

export { router as analysisRouter };
