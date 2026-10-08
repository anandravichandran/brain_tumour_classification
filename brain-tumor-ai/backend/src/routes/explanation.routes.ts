import { Router } from 'express';
import { explanationController } from '../controllers/explanation.controller';

const router = Router();
router.post('/explanation', explanationController);
export { router as explanationRouter };
