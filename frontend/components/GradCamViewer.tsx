'use client';

import React, { useState } from 'react';
import { Eye, Sliders, Download, Sparkles, AlertCircle, RefreshCw } from 'lucide-react';
import { generateGradCam, getFullImageUrl } from '../services/api';

interface GradCamViewerProps {
  analysisId: number;
  originalImageUrl: string;
  initialOverlayUrl?: string;
  diseaseLabels: string[];
  activeDisease?: string;
}

export default function GradCamViewer({
  analysisId,
  originalImageUrl,
  initialOverlayUrl,
  diseaseLabels,
  activeDisease,
}: GradCamViewerProps) {
  const [alpha, setAlpha] = useState<number>(0.5);
  const [selectedDiseaseIdx, setSelectedDiseaseIdx] = useState<number>(0);
  const [overlayUrl, setOverlayUrl] = useState<string | undefined>(initialOverlayUrl);
  const [heatmapUrl, setHeatmapUrl] = useState<string | undefined>(undefined);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<'overlay' | 'side-by-side' | 'heatmap'>('side-by-side');

  const handleUpdateGradCam = async (newIdx?: number, newAlpha?: number) => {
    setIsLoading(true);
    try {
      const idx = newIdx !== undefined ? newIdx : selectedDiseaseIdx;
      const a = newAlpha !== undefined ? newAlpha : alpha;
      const res = await generateGradCam(analysisId, idx, a);
      setOverlayUrl(res.gradcam_overlay_url);
      setHeatmapUrl(res.gradcam_heatmap_url);
    } catch (err) {
      console.error('Failed to generate Grad-CAM:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAlphaChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setAlpha(val);
  };

  const handleDiseaseChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const idx = parseInt(e.target.value, 10);
    setSelectedDiseaseIdx(idx);
    handleUpdateGradCam(idx, alpha);
  };

  const fullOriginalUrl = getFullImageUrl(originalImageUrl);
  const fullOverlayUrl = overlayUrl ? getFullImageUrl(overlayUrl) : fullOriginalUrl;
  const fullHeatmapUrl = heatmapUrl ? getFullImageUrl(heatmapUrl) : null;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
      {/* Title & Controls Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-50 text-teal-600">
              <Eye className="h-4 w-4" />
            </span>
            <h3 className="text-base font-bold text-slate-900">
              Explainable AI — Grad-CAM Saliency Analysis
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Visualizing convolutional feature activation gradients for targeted pathology.
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl">
          <button
            type="button"
            onClick={() => setViewMode('side-by-side')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'side-by-side'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Side-by-Side
          </button>
          <button
            type="button"
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'overlay'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Blended Overlay
          </button>
          <button
            type="button"
            onClick={() => setViewMode('heatmap')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'heatmap'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Heatmap Only
          </button>
        </div>
      </div>

      {/* Target Disease & Transparency Slider Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 items-center bg-slate-50/80 p-4 rounded-xl border border-slate-200">
        {/* Disease Target Selector */}
        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1.5">
            Explain Target Pathology:
          </label>
          <div className="flex gap-2">
            <select
              value={selectedDiseaseIdx}
              onChange={handleDiseaseChange}
              className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
            >
              {diseaseLabels.map((label, idx) => (
                <option key={label} value={idx}>
                  {label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => handleUpdateGradCam(selectedDiseaseIdx, alpha)}
              disabled={isLoading}
              className="px-2.5 py-1.5 rounded-lg border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              title="Re-compute Grad-CAM"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin text-sky-600' : ''}`} />
            </button>
          </div>
        </div>

        {/* Heatmap Transparency Slider */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-700 mb-1.5">
            <span className="flex items-center gap-1.5">
              <Sliders className="h-3.5 w-3.5 text-slate-500" />
              Overlay Transparency (Alpha):
            </span>
            <span className="font-mono text-sky-600">{Math.round(alpha * 100)}% Heatmap</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-slate-600 font-medium">Original</span>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={alpha}
              onChange={handleAlphaChange}
              onMouseUp={() => handleUpdateGradCam(selectedDiseaseIdx, alpha)}
              onTouchEnd={() => handleUpdateGradCam(selectedDiseaseIdx, alpha)}
              className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-600"
            />
            <span className="text-[10px] text-slate-600 font-medium">Full Heatmap</span>
          </div>
        </div>
      </div>

      {/* Visual Canvas Area */}
      <div className="relative">
        {isLoading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-sm rounded-xl">
            <div className="flex items-center gap-2 text-xs font-semibold text-sky-700 bg-white px-4 py-2 rounded-lg shadow-md border border-sky-100">
              <RefreshCw className="h-4 w-4 animate-spin text-sky-600" />
              Computing Grad-CAM gradients...
            </div>
          </div>
        )}

        {viewMode === 'side-by-side' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="text-xs font-bold text-slate-700 flex items-center justify-between">
                <span>Original Chest Radiograph</span>
                <span className="text-[10px] text-slate-600">Standard Frontal AP/PA</span>
              </div>
              <div className="aspect-square rounded-xl bg-slate-950 overflow-hidden border border-slate-200 relative group flex items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={fullOriginalUrl}
                  alt="Original X-ray"
                  className="max-h-full max-w-full object-contain"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs font-bold text-slate-700 flex items-center justify-between">
                <span className="text-teal-700 font-bold flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-teal-500"></span>
                  Grad-CAM Attention Map Overlay
                </span>
                <span className="text-[10px] text-slate-600 font-mono">
                  Target: {diseaseLabels[selectedDiseaseIdx] || 'Finding'}
                </span>
              </div>
              <div className="aspect-square rounded-xl bg-slate-950 overflow-hidden border border-teal-200 relative group flex items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={fullOverlayUrl}
                  alt="Grad-CAM Overlay"
                  className="max-h-full max-w-full object-contain"
                />
              </div>
            </div>
          </div>
        )}

        {viewMode === 'overlay' && (
          <div className="max-w-xl mx-auto space-y-2">
            <div className="text-xs font-bold text-slate-700 text-center">
              Target: <strong className="text-teal-700">{diseaseLabels[selectedDiseaseIdx]}</strong> — Blended Saliency Map
            </div>
            <div className="aspect-square rounded-2xl bg-slate-950 overflow-hidden border border-slate-200 flex items-center justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={fullOverlayUrl}
                alt="Grad-CAM Blended"
                className="max-h-full max-w-full object-contain"
              />
            </div>
          </div>
        )}

        {viewMode === 'heatmap' && (
          <div className="max-w-xl mx-auto space-y-2">
            <div className="text-xs font-bold text-slate-700 text-center">
              Raw Jet Colormap Activation Response
            </div>
            <div className="aspect-square rounded-2xl bg-slate-950 overflow-hidden border border-slate-200 flex items-center justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={fullHeatmapUrl || fullOverlayUrl}
                alt="Grad-CAM Heatmap"
                className="max-h-full max-w-full object-contain"
              />
            </div>
          </div>
        )}
      </div>

      {/* Heatmap Color Interpretation Scale & Action Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-100">
        {/* Colorbar Indicator */}
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-slate-600">Influence:</span>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500">Low (0.0)</span>
            <div className="h-3.5 w-32 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-400 to-red-600 border border-slate-300"></div>
            <span className="text-[10px] font-bold text-red-600">High (1.0)</span>
          </div>
        </div>

        {/* Download Buttons */}
        <div className="flex items-center gap-2">
          {overlayUrl && (
            <a
              href={fullOverlayUrl}
              download={`gradcam_${analysisId}.png`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 shadow-sm transition-colors"
            >
              <Download className="h-3.5 w-3.5 text-slate-500" />
              Download Grad-CAM
            </a>
          )}
        </div>
      </div>

      {/* Interpretability Scientific Disclaimer */}
      <div className="flex items-start gap-2.5 rounded-xl bg-slate-50 border border-slate-200 p-3 text-[11px] text-slate-600 leading-relaxed">
        <AlertCircle className="h-4 w-4 text-sky-600 shrink-0 mt-0.5" />
        <div>
          <strong className="text-slate-800">Interpretability Guidance:</strong> Grad-CAM highlights
          radiographic regions with the strongest gradients flowing into the final convolutional layer.
          These heatmaps indicate <em>model decision salience</em> and are not clinical proof of lesion
          margins or tissue pathology.
        </div>
      </div>
    </div>
  );
}
