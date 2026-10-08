// Wait, I should import axios properly
import axiosInstance from 'axios';

const api = axiosInstance.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 60000,
});

export interface AnalysisResponse {
  success: boolean;
  analysis: {
    prediction: 'tumor' | 'no_tumor';
    confidence: number;
    segmentation: {
      detected: boolean;
      tumor_pixels: number | null;
      tumor_percentage: number | null;
      message: string | null;
    };
    visualizations: {
      original: string;
      mask: string | null;
      overlay: string | null;
      gradcam: string | null;
      gradcam_overlay: string | null;
    };
    model: {
      classifier: string;
      segmenter: string;
      version: string;
    };
  };
  error?: {
    code: string;
    message: string;
  };
}

export const analyzeImage = async (file: File): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post<AnalysisResponse>('/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const getExplanation = async (
  prediction: string, 
  confidence: number, 
  tumorPixels?: number | null, 
  tumorPercentage?: number | null
): Promise<string> => {
  const { data } = await api.post('/explanation', {
    prediction,
    confidence,
    tumorPixels,
    tumorPercentage,
  });
  return data.explanation;
};
