import express from 'express';
import cors from 'cors';
import { env } from './config/env';
import { analysisRouter } from './routes/analysis.routes';
import { explanationRouter } from './routes/explanation.routes';
import { healthRouter } from './routes/health.routes';
import { errorMiddleware } from './middleware/error.middleware';
import { logger } from './utils/logger';

export const app = express();

// CORS — only allow configured frontend
app.use(cors({
  origin: env.frontendUrl.split(',').map(s => s.trim()),
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/', healthRouter);
app.use('/api', analysisRouter);
app.use('/api', explanationRouter);

// 404
app.use((_req, res) => {
  res.status(404).json({ success: false, error: { code: 'NOT_FOUND', message: 'Endpoint not found' } });
});

// Global error handler
app.use(errorMiddleware);

logger.info(`Express app configured | env=${env.nodeEnv}`);
