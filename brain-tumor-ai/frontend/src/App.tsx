import React, { useState, useRef } from 'react';
import { Upload, Activity, Brain, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { analyzeImage, getExplanation, AnalysisResponse } from './services/api';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResponse['analysis'] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      if (selected.size > 10 * 1024 * 1024) {
        setError('File is too large (max 10MB)');
        return;
      }
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
      setExplanation(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setAnalyzing(true);
    setError(null);
    try {
      const res = await analyzeImage(file);
      if (res.success && res.analysis) {
        setResult(res.analysis);

        // Optionally fetch explanation
        setLoadingExplanation(true);
        try {
          const exp = await getExplanation(
            res.analysis.prediction,
            res.analysis.confidence,
            res.analysis.segmentation.tumor_pixels,
            res.analysis.segmentation.tumor_percentage
          );
          setExplanation(exp);
        } catch (e) {
          console.error('Explanation failed', e);
        } finally {
          setLoadingExplanation(false);
        }
      } else {
        setError(res.error?.message || 'Analysis failed');
      }
    } catch (err: any) {
      setError(err.response?.data?.error?.message || err.message || 'An unexpected error occurred');
    } finally {
      setAnalyzing(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setExplanation(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="min-h-screen bg-dark-900 text-slate-200 font-sans selection:bg-primary-500/30">
      {/* Background gradients */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-primary-700/20 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-700/20 blur-[120px]" />
      </div>

      <div className="relative z-10 flex flex-col min-h-screen">
        <header className="px-6 py-4 border-b border-slate-700/50 glass-panel sticky top-0 z-50">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-primary-500/20">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">Neuro<span className="text-primary-400">Scan AI</span></h1>
                <p className="text-xs text-slate-400 font-medium">Academic MRI Analysis Demo</p>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 p-6 flex flex-col items-center">
          <div className="w-full max-w-5xl mx-auto space-y-8 mt-4 animate-fade-in">

            {/* Disclaimer */}


            {!result && (
              <div className="glass-panel rounded-2xl p-8 md:p-12 text-center relative overflow-hidden group transition-all">
                <div className="absolute inset-0 bg-gradient-to-b from-primary-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

                <h2 className="text-3xl font-bold mb-4">Upload MRI Scan</h2>
                <p className="text-slate-400 mb-8 max-w-md mx-auto">
                  Select an MRI image for AI-assisted classification and segmentation analysis. Supported formats: JPG, PNG.
                </p>

                {!preview ? (
                  <div
                    className="border-2 border-dashed border-slate-600 rounded-2xl p-12 hover:border-primary-500 hover:bg-slate-800/50 transition-all cursor-pointer flex flex-col items-center justify-center gap-4 relative z-10"
                    onClick={() => fileInputRef.current?.click()}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        fileInputRef.current?.click();
                      }
                    }}
                  >
                    <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center">
                      <Upload className="w-8 h-8 text-primary-400" />
                    </div>
                    <div className="text-slate-300 font-medium">Click to browse files</div>
                    <div className="text-slate-500 text-sm">Max size 10MB</div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="relative w-64 h-64 mx-auto rounded-2xl overflow-hidden border border-slate-600 shadow-2xl">
                      <img src={preview} alt="MRI Preview" className="w-full h-full object-cover" />
                      {analyzing && (
                        <div className="absolute inset-0 bg-dark-900/80 backdrop-blur-sm flex flex-col items-center justify-center gap-4 text-primary-400">
                          <Activity className="w-10 h-10 animate-pulse-slow" />
                          <span className="font-medium animate-pulse">Analyzing MRI...</span>
                        </div>
                      )}
                    </div>

                    <div className="flex justify-center gap-4">
                      <button
                        onClick={reset}
                        disabled={analyzing}
                        className="px-6 py-2.5 rounded-xl border border-slate-600 text-slate-300 hover:bg-slate-800 transition-colors disabled:opacity-50 font-medium"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleAnalyze}
                        disabled={analyzing}
                        className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-primary-600 to-indigo-600 text-white font-medium hover:from-primary-500 hover:to-indigo-500 transition-all disabled:opacity-50 shadow-lg shadow-primary-500/25 flex items-center gap-2"
                      >
                        {analyzing ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Brain className="w-5 h-5" />}
                        {analyzing ? 'Processing...' : 'Run Analysis'}
                      </button>
                    </div>
                  </div>
                )}

                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept="image/jpeg,image/png,image/jpg"
                  onChange={handleFileSelect}
                />

                {error && (
                  <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-sm">
                    {error}
                  </div>
                )}
              </div>
            )}

            {result && (
              <div className="space-y-6 animate-fade-in">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    <Activity className="text-primary-400" />
                    Analysis Results
                  </h2>
                  <button
                    onClick={reset}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium transition-colors"
                  >
                    <RefreshCw className="w-4 h-4" />
                    New Analysis
                  </button>
                </div>

                {/* Primary metrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="glass-panel rounded-2xl p-6 flex items-center justify-between">
                    <div>
                      <p className="text-slate-400 text-sm font-medium mb-1">Model Prediction</p>
                      <h3 className="text-2xl font-bold flex items-center gap-2">
                        {result.prediction === 'tumor' ? (
                          <span className="text-red-400">Tumor Detected</span>
                        ) : (
                          <span className="text-green-400">No Tumor Detected</span>
                        )}
                      </h3>
                    </div>
                    <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700">
                      {result.prediction === 'tumor' ? <AlertCircle className="text-red-400" /> : <CheckCircle className="text-green-400" />}
                    </div>
                  </div>

                  <div className="glass-panel rounded-2xl p-6 flex items-center justify-between">
                    <div>
                      <p className="text-slate-400 text-sm font-medium mb-1">Confidence Score</p>
                      <h3 className="text-2xl font-bold text-white">
                        {(result.confidence * 100).toFixed(1)}%
                      </h3>
                    </div>
                    <div className="relative w-12 h-12 flex items-center justify-center">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                        <path className="text-slate-700" strokeDasharray="100, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="3" />
                        <path className="text-primary-500" strokeDasharray={`${result.confidence * 100}, 100`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="3" />
                      </svg>
                    </div>
                  </div>
                </div>

                {/* Segmentation Metrics if available */}
                {result.segmentation.detected && (
                  <div className="glass-panel rounded-2xl p-6 flex gap-8">
                    <div>
                      <p className="text-slate-400 text-sm font-medium mb-1">Tumor Pixels</p>
                      <p className="text-xl font-bold text-white">{result.segmentation.tumor_pixels?.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-sm font-medium mb-1">Image Area</p>
                      <p className="text-xl font-bold text-white">{result.segmentation.tumor_percentage?.toFixed(1)}%</p>
                    </div>
                  </div>
                )}

                {/* AI Explanation */}
                <div className="glass-panel rounded-2xl p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Brain className="w-5 h-5 text-indigo-400" />
                    AI Assistant Explanation
                  </h3>
                  {loadingExplanation ? (
                    <div className="flex items-center gap-3 text-slate-400 text-sm">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Generating explanation...
                    </div>
                  ) : explanation ? (
                    <p className="text-slate-300 leading-relaxed text-sm whitespace-pre-wrap">{explanation}</p>
                  ) : (
                    <p className="text-slate-500 text-sm italic">Explanation not available.</p>
                  )}
                </div>

                {/* Visualizations */}
                <h3 className="text-xl font-bold text-white mt-8 mb-4">Visualizations</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">

                  {/* Original */}
                  <div className="glass-panel rounded-xl overflow-hidden flex flex-col">
                    <div className="bg-slate-800/50 p-3 border-b border-slate-700/50 text-sm font-medium text-center">
                      Preprocessed MRI
                    </div>
                    <div className="p-4 flex-1 flex items-center justify-center bg-black">
                      <img src={`data:image/jpeg;base64,${result.visualizations.original}`} alt="Original" className="max-w-full rounded" />
                    </div>
                  </div>

                  {/* GradCAM */}
                  {result.visualizations.gradcam_overlay && (
                    <div className="glass-panel rounded-xl overflow-hidden flex flex-col">
                      <div className="bg-slate-800/50 p-3 border-b border-slate-700/50 text-sm font-medium text-center">
                        Grad-CAM Attention
                      </div>
                      <div className="p-4 flex-1 flex items-center justify-center bg-black">
                        <img src={`data:image/jpeg;base64,${result.visualizations.gradcam_overlay}`} alt="GradCAM" className="max-w-full rounded" />
                      </div>
                    </div>
                  )}

                  {/* Mask */}
                  {result.visualizations.mask && (
                    <div className="glass-panel rounded-xl overflow-hidden flex flex-col">
                      <div className="bg-slate-800/50 p-3 border-b border-slate-700/50 text-sm font-medium text-center">
                        U-Net Mask
                      </div>
                      <div className="p-4 flex-1 flex items-center justify-center bg-black">
                        <img src={`data:image/png;base64,${result.visualizations.mask}`} alt="Mask" className="max-w-full rounded" />
                      </div>
                    </div>
                  )}

                  {/* Overlay */}
                  {result.visualizations.overlay && (
                    <div className="glass-panel rounded-xl overflow-hidden flex flex-col">
                      <div className="bg-slate-800/50 p-3 border-b border-slate-700/50 text-sm font-medium text-center">
                        Tumor Overlay
                      </div>
                      <div className="p-4 flex-1 flex items-center justify-center bg-black">
                        <img src={`data:image/jpeg;base64,${result.visualizations.overlay}`} alt="Overlay" className="max-w-full rounded" />
                      </div>
                    </div>
                  )}

                </div>

                <div className="text-xs text-slate-500 mt-8 text-center pb-8">
                  Models: Classifier ({result.model.classifier}) v{result.model.version} | Segmenter ({result.model.segmenter})
                </div>

              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
