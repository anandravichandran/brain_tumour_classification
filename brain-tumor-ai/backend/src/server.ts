import { app } from './app';
import { env } from './config/env';
import { logger } from './utils/logger';

const server = app.listen(env.port, '0.0.0.0', () => {
  logger.info(`Brain Tumor Backend running on http://0.0.0.0:${env.port}`);
  logger.info(`ML Service: ${env.mlServiceUrl}`);
  logger.info(`Frontend: ${env.frontendUrl}`);
  logger.info(`DeepSeek: ${env.deepseekEnabled ? 'enabled' : 'disabled (no API key)'}`);
});

process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down...');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('unhandledRejection', (reason) => {
  logger.error(`Unhandled rejection: ${reason}`);
});
