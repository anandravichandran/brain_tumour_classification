import { Router } from 'express';
import { healthController, modelInfoController } from '../controllers/health.controller';

const router = Router();
router.get('/health', healthController);
router.get('/model-info', modelInfoController);
export { router as healthRouter };
