'use client';

import React, { useState, useEffect, useRef, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import {
  UploadCloud,
  FileImage,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCcw,
  Download,
  Sliders,
  ShieldCheck,
  Brain,
  Info,
  Clock
} from 'lucide-react';
import {
  fetchModels,
  uploadXrayImage,
  runPredict,
  fetchHistoryDetail,
  getPdfReportUrl,
  getFullImageUrl,
  ModelSummary,
  PredictionResponse,
  DiseasePrediction
} from '../../services/api';
import SampleXrayPicker, { SampleItem } from '../../components/SampleXrayPicker';
import GradCamViewer from '../../components/GradCamViewer';

function UploadPageInner() {
  const searchParams = useSearchParams();
  const existingAnalysisId = searchParams.get('analysis_id');

  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [threshold, setThreshold] = useState<number>(0.5);

  // Upload & File State
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadedFilename, setUploadedFilename] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Results State
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModels(data);
        // Default select first available model with status ready
        const firstReady = data.find((m) => m.status === 'ready');
        if (firstReady) {
          setSelectedModelId(firstReady.id);
        } else if (data.length > 0) {
          setSelectedModelId(data[0].id);
        }
      })
      .catch((err) => console.error('Error fetching models:', err));

    // If an analysis_id was passed in URL query, load its details directly
    if (existingAnalysisId) {
      fetchHistoryDetail(parseInt(existingAnalysisId, 10))
        .then((detail) => {
          setUploadedFilename(detail.image_filename);
          setPreviewUrl(detail.image_url);
          setSelectedModelId(detail.model_id);
          setPredictionResult({
            analysis_id: detail.id,
            image_filename: detail.image_filename,
            image_url: detail.image_url,
            gradcam_url: detail.gradcam_url,
            model_id: detail.model_id,
            model_name: detail.model_name,
            architecture: detail.architecture,
            dataset_name: detail.dataset_name,
            task_type: 'multi_label',
            predictions: detail.predictions,
            top_prediction: detail.top_prediction,
            overall_assessment: detail.overall_assessment,
            processing_time_ms: (detail.processing_time || 0) * 1000,
            timestamp: detail.created_at,
            disclaimer: 'FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.'
          });
        })
        .catch((err) => console.error('Error loading previous analysis:', err));
    }
  }, [existingAnalysisId]);

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = async (selectedFile: File) => {
    setErrorMessage(null);
    const ext = selectedFile.name.split('.').pop()?.toLowerCase();
    if (!['jpg', 'jpeg', 'png'].includes(ext || '')) {
      setErrorMessage('Invalid file format. Please upload JPG, JPEG, or PNG.');
      return;
    }

    if (selectedFile.size > 25 * 1024 * 1024) {
      setErrorMessage('File size exceeds 25MB limit.');
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setPredictionResult(null);

    // Auto upload to server
    setIsUploading(true);
    try {
      const uploadRes = await uploadXrayImage(selectedFile);
      setUploadedFilename(uploadRes.filename);
    } catch (err: any) {
      setErrorMessage(err.message || 'Image upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSelectSample = (sample: SampleItem) => {
    setErrorMessage(null);
    setFile(null);
    setPreviewUrl(`/samples/${sample.filename}`);
    setUploadedFilename(sample.filename);
    setPredictionResult(null);
  };

  const handleClearImage = () => {
    setFile(null);
    setPreviewUrl(null);
    setUploadedFilename(null);
    setPredictionResult(null);
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleExecuteInference = async () => {
    if (!uploadedFilename) {
      setErrorMessage('Please select or upload a chest X-ray image first.');
      return;
    }

    if (!selectedModelId) {
      setErrorMessage('Please select a trained AI model.');
      return;
    }

    const selectedModel = models.find((m) => m.id === selectedModelId);
    if (selectedModel?.status !== 'ready') {
      setErrorMessage(
        `Selected model '${selectedModel?.model_name}' has not been trained yet. Please select an available model with trained weights.`
      );
      return;
    }

    setIsPredicting(true);
    setErrorMessage(null);

    try {
      const result = await runPredict(selectedModelId, uploadedFilename, threshold);
      setPredictionResult(result);
    } catch (err: any) {
      setErrorMessage(err.message || 'Inference execution failed');
    } finally {
      setIsPredicting(false);
    }
  };

  const selectedModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900">
          Chest X-ray AI Screening & Explainability
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Upload an anterior-posterior (AP) or posterior-anterior (PA) radiograph to detect 14 thoracic disease indicators.
        </p>
      </div>

      {/* Error Alert */}
      {errorMessage && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-xs font-semibold text-red-800 flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 text-red-600 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Main Grid: Upload & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Upload & Preview (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-2">
                <FileImage className="h-4 w-4 text-sky-600" />
                Radiograph Image
              </span>
              {previewUrl && (
                <button
                  type="button"
                  onClick={handleClearImage}
                  className="text-xs font-semibold text-slate-500 hover:text-red-600 transition-colors"
                >
                  Remove & Replace
                </button>
              )}
            </div>

            {/* Drop Zone */}
            {!previewUrl ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
                onClick={() => fileInputRef.current?.click()}
                className="cursor-pointer rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50/70 p-8 sm:p-12 text-center hover:border-sky-500 hover:bg-sky-50/20 transition-all"
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".jpg,.jpeg,.png"
                  className="hidden"
                />
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-white shadow-sm border border-slate-200 text-sky-600 mb-3">
                  <UploadCloud className="h-7 w-7" />
                </div>
                <p className="text-sm font-bold text-slate-800">
                  Drag and drop chest X-ray here, or{' '}
                  <span className="text-sky-600 underline">browse files</span>
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Supports standard JPG, JPEG, and PNG formats (up to 25MB)
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="relative aspect-square w-full max-w-md mx-auto rounded-2xl bg-slate-950 overflow-hidden border border-slate-200 flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt="Radiograph preview"
                    className="max-h-full max-w-full object-contain"
                  />
                  {isUploading && (
                    <div className="absolute inset-0 bg-slate-950/60 flex items-center justify-center text-white text-xs font-semibold">
                      Uploading to secure session...
                    </div>
                  )}
                </div>
                <div className="text-center text-xs text-slate-500 font-mono">
                  {uploadedFilename || file?.name}
                </div>
              </div>
            )}
          </div>

          {/* Clinical Sample Selector */}
          <SampleXrayPicker
            onSelectSample={handleSelectSample}
            selectedFilename={uploadedFilename || undefined}
          />
        </div>

        {/* Right Column: Model Selection & Inference Parameters (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Brain className="h-4 w-4 text-teal-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Select Deep Learning Model
                </h3>
              </div>
              <p className="text-xs text-slate-500">
                Only models with verified trained weights are available for inference.
              </p>
            </div>

            {/* Model Options */}
            <div className="space-y-2.5">
              {models.map((m) => {
                const isReady = m.status === 'ready';
                const isSelected = selectedModelId === m.id;
                return (
                  <label
                    key={m.id}
                    className={`flex items-start justify-between p-3.5 rounded-xl border cursor-pointer transition-all ${
                      isSelected
                        ? 'border-sky-500 bg-sky-50/50 ring-2 ring-sky-500/20'
                        : isReady
                        ? 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                        : 'border-slate-200 bg-slate-50 opacity-60 cursor-not-allowed'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <input
                        type="radio"
                        name="model_selection"
                        value={m.id}
                        disabled={!isReady}
                        checked={isSelected}
                        onChange={() => setSelectedModelId(m.id)}
                        className="mt-1 h-4 w-4 text-sky-600 focus:ring-sky-500"
                      />
                      <div>
                        <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                          {m.model_name}
                        </div>
                        <div className="text-[11px] text-slate-500 mt-0.5">
                          Arch: <strong className="text-slate-700">{m.architecture}</strong> • {m.task_type === 'binary' ? 'Binary Classification' : '14 Multi-label'}
                        </div>
                        <div className="text-[10px] text-slate-400 mt-0.5">
                          Dataset: {m.dataset_name}
                        </div>
                      </div>
                    </div>
                    <div>
                      {isReady ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                          Ready
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-200 text-slate-500">
                          Unloaded
                        </span>
                      )}
                    </div>
                  </label>
                );
              })}
            </div>

            {/* Decision Threshold Slider */}
            <div className="pt-2 border-t border-slate-100 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                <span className="flex items-center gap-1.5">
                  <Sliders className="h-3.5 w-3.5 text-slate-500" />
                  Decision Probability Threshold:
                </span>
                <span className="font-mono text-sky-600 font-bold">{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-600"
              />
              <div className="flex justify-between text-[10px] text-slate-600">
                <span>Higher Sensitivity (10%)</span>
                <span>Standard (50%)</span>
                <span>Higher Specificity (90%)</span>
              </div>
            </div>

            {/* Inference Execution CTA */}
            <button
              type="button"
              onClick={handleExecuteInference}
              disabled={isPredicting || !uploadedFilename || selectedModel?.status !== 'ready'}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-sky-600 py-3.5 text-sm font-bold text-white shadow-md shadow-sky-600/20 hover:bg-sky-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isPredicting ? (
                <>
                  <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Running Neural Network Inference...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-white" />
                  Run AI Disease Analysis
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Prediction Results & Explainability Section */}
      {predictionResult && (
        <div className="space-y-6 pt-4 border-t border-slate-200 animate-fadeIn">
          {/* Result Header & PDF Export */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-600">
                  Screening Assessment #{predictionResult.analysis_id}
                </span>
                <span className="text-xs text-slate-400">•</span>
                <span className="flex items-center gap-1 text-xs text-slate-500 font-mono">
                  <Clock className="h-3.5 w-3.5" />
                  {predictionResult.processing_time_ms} ms
                </span>
              </div>
              <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 mt-1">
                {predictionResult.overall_assessment}
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Model: <strong>{predictionResult.model_name}</strong> ({predictionResult.architecture}) • Trained on: {predictionResult.dataset_name}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <a
                href={getPdfReportUrl(predictionResult.analysis_id)}
                download={`medvision_report_${predictionResult.analysis_id}.pdf`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-bold text-white shadow hover:bg-slate-800 transition-colors"
              >
                <Download className="h-4 w-4" />
                Download PDF Medical Report
              </a>
            </div>
          </div>

          {/* Interactive Grad-CAM Explainability Viewer */}
          <GradCamViewer
            analysisId={predictionResult.analysis_id}
            originalImageUrl={predictionResult.image_url}
            initialOverlayUrl={predictionResult.gradcam_url}
            diseaseLabels={predictionResult.predictions.map((p) => p.disease)}
            activeDisease={predictionResult.top_prediction?.disease}
          />

          {/* Detailed Predictions Table */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Comprehensive Pathology Detection Scores
                </h3>
                <p className="text-xs text-slate-500">
                  Sigmoid output probabilities compared against decision threshold ({(threshold * 100).toFixed(0)}%)
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/80 text-slate-600 font-semibold uppercase tracking-wider text-[11px]">
                    <th className="py-3 px-4">Condition / Pathology</th>
                    <th className="py-3 px-4">Probability Score</th>
                    <th className="py-3 px-4">Confidence Bar</th>
                    <th className="py-3 px-4">Threshold</th>
                    <th className="py-3 px-4">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {predictionResult.predictions.map((pred) => (
                    <tr
                      key={pred.disease}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        pred.is_positive ? 'bg-red-50/30' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-bold text-slate-900">
                        {pred.disease}
                      </td>
                      <td className="py-3 px-4 font-mono font-semibold text-slate-800">
                        {(pred.score * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 w-48">
                        <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              pred.is_positive
                                ? 'bg-red-500'
                                : pred.score >= 0.25
                                ? 'bg-amber-400'
                                : 'bg-sky-400'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(3, pred.score * 100))}%` }}
                          />
                        </div>
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-500">
                        {(pred.threshold * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 px-4">
                        {pred.is_positive ? (
                          <span className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[10px] font-bold bg-red-100 text-red-800">
                            ELEVATED FINDING
                          </span>
                        ) : (
                          <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold bg-slate-100 text-slate-600">
                            Within Limits
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function UploadPage() {
  return (
    <Suspense fallback={
      <div className="p-8 text-center text-xs text-slate-500">
        Loading MedVision AI Analysis Interface...
      </div>
    }>
      <UploadPageInner />
    </Suspense>
  );
}
