'use client';

import React, { useEffect, useState } from 'react';
import {
  Database,
  ExternalLink,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Layers,
  CheckCircle2,
  Lock
} from 'lucide-react';
import { fetchDatasets, DatasetInfo } from '../../services/api';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  useEffect(() => {
    fetchDatasets().then(setDatasets).catch(console.error);
  }, []);

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 flex items-center gap-2.5">
          <Database className="h-7 w-7 text-sky-600" />
          Strict Dataset Provenance & Specifications
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Per MedVision AI specification, ONLY these four medical imaging datasets are supported with no silent merging.
        </p>
      </div>

      {/* Dataset Cards */}
      <div className="space-y-6">
        {datasets.map((dataset, idx) => (
          <div
            key={dataset.slug}
            className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4"
          >
            {/* Top row */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-sky-100 text-sky-700 font-bold text-xs">
                    0{idx + 1}
                  </span>
                  <h2 className="text-lg font-bold text-slate-900">{dataset.name}</h2>
                </div>
                <p className="text-xs text-slate-500 mt-1">{dataset.description}</p>
              </div>

              <div className="flex items-center gap-2">
                <a
                  href={dataset.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-sky-700 hover:bg-sky-50 transition-colors"
                >
                  Official Source <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>
            </div>

            {/* Grid Specs */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1">
                <span className="font-bold text-slate-500 uppercase text-[10px] tracking-wider block">
                  Task Specification
                </span>
                <div className="font-semibold text-slate-800">{dataset.task_type}</div>
                <div className="text-slate-500 text-[11px]">{dataset.total_images_note}</div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1">
                <span className="font-bold text-slate-500 uppercase text-[10px] tracking-wider block">
                  Patient Leakage Prevention
                </span>
                <div className="font-semibold text-teal-700 flex items-center gap-1">
                  <ShieldCheck className="h-4 w-4" /> Strictly Partitioned
                </div>
                <div className="text-slate-500 text-[11px]">
                  All studies from the same Patient ID are kept within one split.
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1">
                <span className="font-bold text-slate-500 uppercase text-[10px] tracking-wider block">
                  Licensing & Provenance
                </span>
                <div className="font-semibold text-slate-800">{dataset.license}</div>
                <div className="text-slate-500 text-[11px]">{dataset.uncertainty_handling}</div>
              </div>
            </div>

            {/* Disease Classes Badges */}
            <div>
              <span className="text-xs font-bold text-slate-700 block mb-2">
                Pathology Classes ({dataset.classes.length}):
              </span>
              <div className="flex flex-wrap gap-1.5">
                {dataset.classes.map((clsName) => (
                  <span
                    key={clsName}
                    className="inline-flex items-center px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-100 text-slate-700 border border-slate-200"
                  >
                    {clsName}
                  </span>
                ))}
              </div>
            </div>

            {/* Provenance note */}
            <div className="text-[11px] text-slate-500 leading-relaxed pt-2 border-t border-slate-100">
              <strong>Provenance Details:</strong> {dataset.provenance}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
