import multer from 'multer';
import { env } from '../config/env';

// Use memory storage — we stream to Python, never persist on Node
const storage = multer.memoryStorage();

const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png'];

export const uploadMiddleware = multer({
  storage,
  limits: {
    fileSize: env.maxFileSizeBytes,
    files: 1,
  },
  fileFilter: (_req, file, cb) => {
    if (ALLOWED_MIME_TYPES.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${file.mimetype}. Allowed: image/jpeg, image/png`));
    }
  },
}).single('file');
