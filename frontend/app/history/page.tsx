'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  History as HistoryIcon,
  Search,
  Filter,
  Trash2,
  Download,
  ExternalLink,
  Clock,
  CheckCircle2,
  AlertTriangle,
  RotateCcw
} from 'lucide-react';
import {
  fetchHistory,
  deleteHistoryItem,
  fetchModels,
  getPdfReportUrl,
  getFullImageUrl,
  HistoryItem,
  ModelSummary
} from '../../services/api';

export default function HistoryPage() {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [search, setSearch] = useState<string>('');
  const [selectedModelId, setSelectedModelId] = useState<number | undefined>(undefined);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const loadHistory = async () => {
    setIsLoading(true);
    try {
      const res = await fetchHistory(search, selectedModelId);
      setHistoryItems(res.items);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error);
  }, []);

  useEffect(() => {
    loadHistory();
  }, [selectedModelId]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadHistory();
  };

  const handleDelete = async (id: number) => {
    if (!confirm(`Are you sure you want to delete analysis record #${id}?`)) return;
    setDeletingId(id);
    try {
      await deleteHistoryItem(id);
      setHistoryItems((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      alert('Failed to delete history item');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 flex items-center gap-2.5">
            <HistoryIcon className="h-7 w-7 text-sky-600" />
            Prediction & Screening History
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Audit log of chest radiographs analyzed with deep learning model predictions and Grad-CAM explanations.
          </p>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm flex flex-col md:flex-row gap-3 items-center justify-between">
        <form onSubmit={handleSearchSubmit} className="flex-1 w-full md:max-w-md relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search by image filename..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-xl border border-slate-200 bg-slate-50/70 pl-10 pr-4 py-2 text-xs font-medium text-slate-800 placeholder-slate-400 focus:border-sky-500 focus:bg-white focus:outline-none focus:ring-1 focus:ring-sky-500"
          />
        </form>

        <div className="flex items-center gap-2 w-full md:w-auto">
          <Filter className="h-4 w-4 text-slate-400 shrink-0" />
          <select
            value={selectedModelId || ''}
            onChange={(e) =>
              setSelectedModelId(e.target.value ? parseInt(e.target.value, 10) : undefined)
            }
            className="w-full md:w-auto rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm focus:border-sky-500 focus:outline-none"
          >
            <option value="">All Models</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.model_name}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={loadHistory}
            className="p-2 rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
            title="Refresh history"
          >
            <RotateCcw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* History Records Table */}
      <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        {historyItems.length === 0 ? (
          <div className="text-center py-16 px-4">
            <HistoryIcon className="h-10 w-10 text-slate-300 mx-auto mb-2" />
            <h3 className="text-sm font-bold text-slate-800">No Records Found</h3>
            <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
              No historical analyses match your query. Upload an X-ray to generate new analysis records.
            </p>
            <Link
              href="/upload"
              className="mt-4 inline-flex items-center gap-1.5 rounded-xl bg-sky-600 px-4 py-2 text-xs font-semibold text-white shadow hover:bg-sky-700"
            >
              Analyze New Image
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold uppercase tracking-wider text-[11px]">
                  <th className="py-3.5 px-4">Analysis</th>
                  <th className="py-3.5 px-4">Model & Architecture</th>
                  <th className="py-3.5 px-4">Primary Assessment</th>
                  <th className="py-3.5 px-4">Latency</th>
                  <th className="py-3.5 px-4">Date & Time</th>
                  <th className="py-3.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {historyItems.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/70 transition-colors">
                    {/* Thumbnail & Filename */}
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-3">
                        <div className="h-11 w-11 rounded-lg bg-slate-950 overflow-hidden flex-shrink-0 border border-slate-200">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={getFullImageUrl(item.image_url)}
                            alt="X-ray thumbnail"
                            className="h-full w-full object-cover"
                          />
                        </div>
                        <div>
                          <div className="font-bold text-slate-900 font-mono text-[11px] truncate max-w-[150px]">
                            #{item.id} • {item.image_filename}
                          </div>
                          <span className="text-[10px] text-slate-400">Radiograph AP/PA</span>
                        </div>
                      </div>
                    </td>

                    {/* Model & Dataset */}
                    <td className="py-3.5 px-4">
                      <div className="font-bold text-slate-800">{item.model_name}</div>
                      <div className="text-[10px] text-slate-500">{item.dataset_name}</div>
                    </td>

                    {/* Overall Assessment */}
                    <td className="py-3.5 px-4">
                      <div className="font-semibold text-slate-900 max-w-xs truncate">
                        {item.overall_assessment}
                      </div>
                      {item.top_prediction && (
                        <div className="text-[10px] text-slate-500">
                          Top score: <strong>{item.top_prediction.disease}</strong> ({Math.round(item.top_prediction.score * 100)}%)
                        </div>
                      )}
                    </td>

                    {/* Latency */}
                    <td className="py-3.5 px-4 font-mono text-slate-600">
                      {(item.processing_time * 1000).toFixed(0)} ms
                    </td>

                    {/* Timestamp */}
                    <td className="py-3.5 px-4 text-slate-500">
                      {new Date(item.created_at).toLocaleString()}
                    </td>

                    {/* Action Buttons */}
                    <td className="py-3.5 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <Link
                          href={`/upload?analysis_id=${item.id}`}
                          className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:text-sky-600 hover:bg-slate-50"
                          title="Open Analysis & Grad-CAM"
                        >
                          <ExternalLink className="h-3.5 w-3.5" />
                        </Link>
                        <a
                          href={getPdfReportUrl(item.id)}
                          download={`medvision_report_${item.id}.pdf`}
                          target="_blank"
                          rel="noreferrer"
                          className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:text-sky-600 hover:bg-slate-50"
                          title="Download PDF Medical Report"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </a>
                        <button
                          type="button"
                          onClick={() => handleDelete(item.id)}
                          disabled={deletingId === item.id}
                          className="p-1.5 rounded-lg border border-slate-200 text-slate-400 hover:text-red-600 hover:bg-red-50 disabled:opacity-50"
                          title="Delete Record"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
