import dotenv from 'dotenv';
dotenv.config();

function requireEnv(key: string): string {
  const val = process.env[key];
  if (!val) throw new Error(`Missing required environment variable: ${key}`);
  return val;
}

function optionalEnv(key: string, defaultValue: string): string {
  return process.env[key] ?? defaultValue;
}

export const env = {
  port: parseInt(optionalEnv('PORT', '3001'), 10),
  mlServiceUrl: optionalEnv('ML_SERVICE_URL', 'http://localhost:8000'),
  deepseekApiKey: process.env['DEEPSEEK_API_KEY'] ?? '',
  deepseekBaseUrl: optionalEnv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
  deepseekModel: optionalEnv('DEEPSEEK_MODEL', 'deepseek-chat'),
  maxFileSizeMb: parseInt(optionalEnv('MAX_FILE_SIZE_MB', '10'), 10),
  frontendUrl: optionalEnv('FRONTEND_URL', 'http://localhost:5173'),
  nodeEnv: optionalEnv('NODE_ENV', 'development'),

  get maxFileSizeBytes(): number {
    return this.maxFileSizeMb * 1024 * 1024;
  },

  get deepseekEnabled(): boolean {
    return Boolean(this.deepseekApiKey);
  },
} as const;
