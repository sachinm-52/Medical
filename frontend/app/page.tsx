'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Activity,
  UploadCloud,
  CheckCircle2,
  Layers,
  Eye,
  Database,
  ArrowRight,
  ShieldCheck,
  Brain,
  FileSpreadsheet,
  AlertTriangle,
  History as HistoryIcon
} from 'lucide-react';
import { fetchModels, fetchDatasets, fetchHistory, ModelSummary, DatasetInfo, HistoryItem } from '../services/api';

export default function DashboardPage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetchModels().catch(() => []),
      fetchDatasets().catch(() => []),
      fetchHistory('', undefined).then((res) => res.items).catch(() => []),
    ]).then(([modelsData, datasetsData, historyData]) => {
      setModels(modelsData);
      setDatasets(datasetsData);
      setHistory(historyData);
      setIsLoading(false);
    });
  }, []);

  const readyModels = models.filter((m) => m.status === 'ready');

  return (
    <div className="space-y-8 pb-12">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl border border-slate-200 bg-gradient-to-br from-white via-sky-50/40 to-teal-50/30 p-8 sm:p-10 shadow-sm">
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white/80 px-3 py-1 text-xs font-semibold text-sky-800 shadow-sm">
            <span className="flex h-2 w-2 rounded-full bg-sky-500 animate-pulse"></span>
            Explainable AI • Chest Radiography Research Platform
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-slate-900 leading-tight">
            Interpretable Multi-Disease Detection with <span className="text-sky-600">Grad-CAM</span>
          </h1>

          <p className="text-sm sm:text-base text-slate-600 leading-relaxed">
            MedVision AI evaluates frontal chest radiographs using deep convolutional backbones
            (DenseNet121, ResNet50, EfficientNet-B0) across 14 thoracic disease categories,
            pairing probabilistic scores with transparent spatial activation heatmaps.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <Link
              href="/upload"
              className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-5 py-3 text-sm font-semibold text-white shadow-md shadow-sky-600/20 hover:bg-sky-700 transition-all hover:gap-3"
            >
              <UploadCloud className="h-4 w-4" />
              Upload & Analyze Chest X-ray
              <ArrowRight className="h-4 w-4" />
            </Link>

            <Link
              href="/models"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-all"
            >
              <Layers className="h-4 w-4 text-slate-500" />
              View Benchmark Metrics
            </Link>
          </div>
        </div>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Total Analyses</span>
            <Activity className="h-4 w-4 text-sky-500" />
          </div>
          <div className="text-2xl font-black text-slate-900">
            {history.length}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Real evaluations logged in DB
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Ready AI Models</span>
            <Brain className="h-4 w-4 text-teal-500" />
          </div>
          <div className="text-2xl font-black text-slate-900">
            {readyModels.length} <span className="text-xs text-slate-400 font-normal">/ {models.length} registered</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Verified trained weights loaded
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Supported Datasets</span>
            <Database className="h-4 w-4 text-indigo-500" />
          </div>
          <div className="text-2xl font-black text-slate-900">
            4 Strict Datasets
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Kaggle, NIH14, CheXpert, MedMNIST
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Explainability</span>
            <Eye className="h-4 w-4 text-emerald-500" />
          </div>
          <div className="text-2xl font-black text-slate-900">
            Grad-CAM
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Layer4 / Norm5 Saliency Mapping
          </div>
        </div>
      </div>

      {/* How the AI Analysis Works Section */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
        <div>
          <h2 className="text-lg font-bold text-slate-900">How MedVision AI Analyzes Radiographs</h2>
          <p className="text-xs text-slate-500 mt-0.5">End-to-end clinical deep learning workflow with strict provenance</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl border border-slate-100 bg-slate-50/70 space-y-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-100 text-sky-700 font-bold text-xs">
              01
            </div>
            <h3 className="text-sm font-bold text-slate-800">Radiograph Ingestion</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Standardized format validation (JPEG/PNG), resolution normalization (224x224), and ImageNet channel standardization.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-100 bg-slate-50/70 space-y-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-100 text-sky-700 font-bold text-xs">
              02
            </div>
            <h3 className="text-sm font-bold text-slate-800">Deep Learning Inference</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Feed-forward pass through DenseNet121, ResNet50, or EfficientNet-B0 with Sigmoid activations for independent multi-label probabilities.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-100 bg-slate-50/70 space-y-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-100 text-teal-700 font-bold text-xs">
              03
            </div>
            <h3 className="text-sm font-bold text-slate-800">Grad-CAM Attribution</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Backpropagation of target class logits into the final convolutional layer generates spatial gradient heatmaps highlighting influential lung fields.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-100 bg-slate-50/70 space-y-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-100 text-teal-700 font-bold text-xs">
              04
            </div>
            <h3 className="text-sm font-bold text-slate-800">Clinical Export</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Structured confidence scores, normal vs abnormal screening assessment, and instant downloadable PDF radiology summary report.
            </p>
          </div>
        </div>
      </div>

      {/* Two Column Grid: Supported Models & Recent Analyses */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Available Models */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-slate-900">Supported AI Models</h2>
              <p className="text-xs text-slate-500">Only models with verified trained weights are marked ready</p>
            </div>
            <Link href="/models" className="text-xs font-semibold text-sky-600 hover:text-sky-700 flex items-center gap-1">
              All Metrics <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <div className="space-y-3">
            {models.map((m) => (
              <div
                key={m.id}
                className="flex items-center justify-between p-3.5 rounded-xl border border-slate-200 bg-slate-50/50 hover:bg-slate-50 transition-colors"
              >
                <div>
                  <div className="text-xs font-bold text-slate-900 flex items-center gap-2">
                    {m.model_name}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5">
                    Dataset: <strong className="text-slate-700">{m.dataset_name}</strong> • Arch: {m.architecture}
                  </div>
                </div>
                <div>
                  {m.status === 'ready' ? (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                      <CheckCircle2 className="h-3 w-3 text-emerald-600" /> Ready
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-semibold bg-slate-100 text-slate-500 border border-slate-200">
                      Not Trained
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Analyses History */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-slate-900">Recent Analyses</h2>
              <p className="text-xs text-slate-500">Evaluations stored in database records</p>
            </div>
            <Link href="/history" className="text-xs font-semibold text-sky-600 hover:text-sky-700 flex items-center gap-1">
              Full History <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          {history.length === 0 ? (
            <div className="text-center py-8 rounded-xl border border-dashed border-slate-200 bg-slate-50/50">
              <HistoryIcon className="h-8 w-8 text-slate-400 mx-auto mb-2" />
              <p className="text-xs font-bold text-slate-700">No Analysis Records Yet</p>
              <p className="text-[11px] text-slate-500 mt-1 max-w-xs mx-auto">
                No simulated data is shown. Upload your first chest X-ray to log real predictions.
              </p>
              <Link
                href="/upload"
                className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-sky-700 shadow-sm"
              >
                Start First Analysis
              </Link>
            </div>
          ) : (
            <div className="space-y-2.5">
              {history.slice(0, 4).map((item) => (
                <div
                  key={item.id}
                  className="flex items-center justify-between p-3 rounded-xl border border-slate-200 hover:border-slate-300 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-slate-900 overflow-hidden flex-shrink-0">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={item.image_url}
                        alt="X-ray thumbnail"
                        className="h-full w-full object-cover"
                      />
                    </div>
                    <div>
                      <div className="text-xs font-bold text-slate-900 truncate max-w-[180px] sm:max-w-xs">
                        {item.overall_assessment}
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {item.model_name} • {new Date(item.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                  <Link
                    href={`/upload?analysis_id=${item.id}`}
                    className="text-xs font-semibold text-sky-600 hover:text-sky-700"
                  >
                    View
                  </Link>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Dataset Provenance Summary */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900">Strict Dataset Provenance</h2>
            <p className="text-xs text-slate-500">
              Only four validated medical imaging datasets are supported with no silent merging or patient leakage.
            </p>
          </div>
          <Link href="/datasets" className="text-xs font-semibold text-sky-600 hover:text-sky-700 flex items-center gap-1">
            Dataset Specs <ArrowRight className="h-3 w-3" />
          </Link>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {datasets.map((d) => (
            <div key={d.slug} className="p-4 rounded-xl border border-slate-200 bg-slate-50/40 space-y-2">
              <div className="text-xs font-bold text-slate-900">{d.name}</div>
              <div className="text-[11px] text-slate-600 font-medium">{d.task_type}</div>
              <p className="text-[10px] text-slate-500 line-clamp-3 leading-relaxed">
                {d.description}
              </p>
              <div className="pt-1 flex items-center justify-between text-[10px]">
                <span className="font-semibold text-teal-700">Patient-level Split</span>
                <span className="text-slate-400 font-mono">{d.num_classes} Classes</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
