'use client';

import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  TrendingUp,
  Award,
  Layers,
  Database,
  CheckCircle2,
  AlertCircle,
  FileSpreadsheet,
  Cpu
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  LineChart,
  Line,
  CartesianGrid
} from 'recharts';
import { fetchModels, fetchMetrics, ModelSummary, MetricItem } from '../../services/api';

export default function ModelPerformancePage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<MetricItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchModels().then((data) => {
      setModels(data);
      const readyModel = data.find((m) => m.status === 'ready');
      if (readyModel) {
        setSelectedModelId(readyModel.id);
      } else if (data.length > 0) {
        setSelectedModelId(data[0].id);
      }
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      setIsLoading(true);
      fetchMetrics(selectedModelId)
        .then((data) => setMetrics(data))
        .catch(console.error)
        .finally(() => setIsLoading(false));
    }
  }, [selectedModelId]);

  const selectedModel = models.find((m) => m.id === selectedModelId);

  // Extract metrics from list
  const getMetricVal = (name: string) => {
    const item = metrics.find((m) => m.metric_name === name);
    return item?.metric_value !== undefined && item?.metric_value !== null ? item.metric_value : null;
  };

  const perClassMetric = metrics.find((m) => m.metric_name === 'per_class')?.metric_data || {};
  const historyMetric = metrics.find((m) => m.metric_name === 'training_history')?.metric_data || {};

  // Format per-class data for Recharts Bar Chart
  const perClassChartData = Object.entries(perClassMetric).map(([disease, data]: [string, any]) => ({
    disease: disease.length > 11 ? disease.slice(0, 10) + '..' : disease,
    fullName: disease,
    rocAuc: data.roc_auc ? +(data.roc_auc * 100).toFixed(1) : 0,
    f1Score: data.f1_score ? +(data.f1_score * 100).toFixed(1) : 0,
    prAuc: data.pr_auc ? +(data.pr_auc * 100).toFixed(1) : 0,
    support: data.support || 0
  }));

  // Format training curves for LineChart
  const lossCurveData = (historyMetric.epochs || []).map((ep: number, idx: number) => ({
    epoch: `Ep ${ep}`,
    trainLoss: historyMetric.train_loss ? historyMetric.train_loss[idx] : 0,
    valLoss: historyMetric.val_loss ? historyMetric.val_loss[idx] : 0,
  }));

  const rocAuc = getMetricVal('roc_auc');
  const prAuc = getMetricVal('pr_auc');
  const f1 = getMetricVal('f1_score');
  const accuracy = getMetricVal('accuracy');
  const precision = getMetricVal('precision');
  const recall = getMetricVal('recall');

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 flex items-center gap-2.5">
          <BarChart3 className="h-7 w-7 text-sky-600" />
          Model Performance & Benchmark Metrics
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Empirical evaluation results calculated on strictly isolated patient-level test partitions.
        </p>
      </div>

      {/* Model Selector Bar */}
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm flex flex-wrap items-center gap-2">
        <span className="text-xs font-bold text-slate-700 mr-2">Select Architecture:</span>
        {models.map((m) => {
          const isSelected = selectedModelId === m.id;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => setSelectedModelId(m.id)}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                isSelected
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              {m.architecture} ({m.dataset_name})
            </button>
          );
        })}
      </div>

      {/* Model Overview Card */}
      {selectedModel && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-slate-900">{selectedModel.model_name}</h2>
                {selectedModel.status === 'ready' ? (
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-700 border border-emerald-200">
                    <CheckCircle2 className="h-3 w-3" /> Ready
                  </span>
                ) : (
                  <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-500">
                    Pending Training
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                Backbone: <strong>{selectedModel.architecture}</strong> • Dataset: <strong>{selectedModel.dataset_name}</strong> • Classes: {selectedModel.num_classes}
              </p>
            </div>
            <div className="text-xs text-slate-500 font-mono">
              Split: <strong className="text-slate-800">Patient-Level Test Cohort</strong>
            </div>
          </div>

          {/* Primary Metric Score Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-2">
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">ROC-AUC (Macro)</div>
              <div className="text-xl font-black text-sky-700 font-mono">
                {rocAuc !== null ? (rocAuc * 100).toFixed(1) + '%' : '—'}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">PR-AUC</div>
              <div className="text-xl font-black text-teal-700 font-mono">
                {prAuc !== null ? (prAuc * 100).toFixed(1) + '%' : '—'}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">F1-Score</div>
              <div className="text-xl font-black text-indigo-700 font-mono">
                {f1 !== null ? (f1 * 100).toFixed(1) + '%' : '—'}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">Accuracy</div>
              <div className="text-xl font-black text-slate-800 font-mono">
                {accuracy !== null ? (accuracy * 100).toFixed(1) + '%' : '—'}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">Precision</div>
              <div className="text-xl font-black text-slate-800 font-mono">
                {precision !== null ? (precision * 100).toFixed(1) + '%' : '—'}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">Recall</div>
              <div className="text-xl font-black text-slate-800 font-mono">
                {recall !== null ? (recall * 100).toFixed(1) + '%' : '—'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recharts Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Per-Class ROC-AUC & F1 Bar Chart (8 cols) */}
        <div className="lg:col-span-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-slate-900">Per-Class Discrimination Metrics</h3>
              <p className="text-xs text-slate-500">ROC-AUC and F1 scores per thoracic condition</p>
            </div>
          </div>

          {perClassChartData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-xs text-slate-400">
              No experimental evaluation metrics loaded for this model.
            </div>
          ) : (
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={perClassChartData} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="disease" angle={-45} textAnchor="end" tick={{ fontSize: 10 }} interval={0} />
                  <YAxis tick={{ fontSize: 10 }} domain={[0, 100]} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-lg text-xs space-y-1">
                            <div className="font-bold text-slate-900">{d.fullName}</div>
                            <div className="text-sky-700">ROC-AUC: <strong>{d.rocAuc}%</strong></div>
                            <div className="text-teal-700">F1-Score: <strong>{d.f1Score}%</strong></div>
                            <div className="text-slate-500">Test Support: {d.support} samples</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px', fontSize: '11px' }} />
                  <Bar dataKey="rocAuc" name="ROC-AUC (%)" fill="#0284c7" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="f1Score" name="F1-Score (%)" fill="#0d9488" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Training & Validation Loss Curves (4 cols) */}
        <div className="lg:col-span-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <div>
            <h3 className="text-base font-bold text-slate-900">Training Loss Dynamics</h3>
            <p className="text-xs text-slate-500">BCEWithLogitsLoss over training epochs</p>
          </div>

          {lossCurveData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-xs text-slate-400">
              No training loss history recorded for this model.
            </div>
          ) : (
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossCurveData} margin={{ top: 10, right: 10, left: -25, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#0284c7" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#e11d48" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Backbone Architecture Comparison Section */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Deep Learning Architecture Comparison</h3>
          <p className="text-xs text-slate-500">Comparative characteristics of DenseNet121, ResNet50, and EfficientNet-B0</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl border border-sky-200 bg-sky-50/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-sky-800">DenseNet121 (Primary)</span>
              <span className="text-[10px] font-mono bg-sky-100 text-sky-800 px-2 py-0.5 rounded">7.98M params</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Densely connected convolutional blocks encourage feature reuse and prevent gradient vanishing. Highest radiology literature adoption.
            </p>
            <div className="text-[11px] text-slate-500 pt-1">
              Grad-CAM Target: <code className="bg-white px-1.5 py-0.5 rounded border text-slate-700">features.norm5</code>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-800">ResNet50 (Residual)</span>
              <span className="text-[10px] font-mono bg-slate-200 text-slate-700 px-2 py-0.5 rounded">25.56M params</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Standard residual bottleneck skip-connections for deep representation learning. Widely used comparative baseline.
            </p>
            <div className="text-[11px] text-slate-500 pt-1">
              Grad-CAM Target: <code className="bg-white px-1.5 py-0.5 rounded border text-slate-700">layer4[-1]</code>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-teal-200 bg-teal-50/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-teal-800">EfficientNet-B0 (Scaled)</span>
              <span className="text-[10px] font-mono bg-teal-100 text-teal-800 px-2 py-0.5 rounded">5.29M params</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Compound scaling of network depth, width, and input resolution. Optimal for resource-constrained edge deployments.
            </p>
            <div className="text-[11px] text-slate-500 pt-1">
              Grad-CAM Target: <code className="bg-white px-1.5 py-0.5 rounded border text-slate-700">features[-1]</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
