# Deployment Guide

## Vercel Deployment (Frontend)
1. Link your GitHub repository to Vercel.
2. Select the `frontend/` directory as the root.
3. Framework Preset: `Vite`
4. Build Command: `npm run build`
5. Output Directory: `dist`
6. Add Environment Variables:
   - `VITE_API_URL`: Your deployed Render Node backend URL + `/api` (e.g., `https://brain-backend.onrender.com/api`)

## Render Deployment (Backend & ML Service)

You will deploy two separate Web Services on Render.

### 1. ML Service (Python)
1. **Root Directory**: `ml-service/`
2. **Environment**: `Docker`
3. Add Environment Variables:
   - `PORT`: `8000`
   - `DEVICE`: `cpu`
   - `MAX_FILE_SIZE_MB`: `10`
   - `ALLOWED_ORIGINS`: Your Vercel frontend URL + Render backend URL

### 2. API Gateway (Node.js)
1. **Root Directory**: `backend/`
2. **Environment**: `Node`
3. **Build Command**: `npm ci && npm run build`
4. **Start Command**: `npm start`
5. Add Environment Variables:
   - `ML_SERVICE_URL`: URL of your deployed ML Service on Render
   - `FRONTEND_URL`: URL of your deployed Vercel Frontend
   - `DEEPSEEK_API_KEY`: (Optional) Your DeepSeek API Key
