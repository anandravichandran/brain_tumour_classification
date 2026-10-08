# Troubleshooting

## Model Loading Errors
If the ML service fails to start or `/health` reports models as unavailable:
- Ensure the `vgg16_brain_tumor.h5` is present in `ml-service/models/classification/`.
- Ensure it is a valid Keras HDF5 model.
- If segmentation fails gracefully, verify `unet_brain_tumor.h5` exists in `ml-service/models/segmentation/`.

## DeepSeek Integration Not Working
- Ensure `DEEPSEEK_API_KEY` is set in the backend `.env` file or environment variables.
- The system will still return valid ML inference results even if DeepSeek is offline or unconfigured.

## File Upload Issues
- `413 Payload Too Large`: The file exceeds `MAX_FILE_SIZE_MB` (default 10MB).
- `415 Unsupported Media Type`: Only `image/jpeg` and `image/png` are accepted.

## Cross-Origin Resource Sharing (CORS)
- If the frontend cannot communicate with the backend, verify the `FRONTEND_URL` environment variable on the Node server matches your deployed Vercel frontend URL exactly (including `https://` and excluding trailing slashes).
