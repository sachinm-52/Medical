'use client';

import React from 'react';
import Image from 'next/image';
import { Sparkles, CheckCircle2 } from 'lucide-react';

export interface SampleItem {
  key: string;
  filename: string;
  title: string;
  description: string;
  category: 'Normal' | 'Pneumonia' | 'Cardiomegaly' | 'Effusion' | 'Complex';
}

const SAMPLE_XRAYS: SampleItem[] = [
  {
    key: 'normal',
    filename: 'sample_normal.png',
    title: 'Normal Anatomy',
    description: 'Clear lung fields, sharp costophrenic angles, normal cardiothoracic ratio.',
    category: 'Normal'
  },
  {
    key: 'cardiomegaly',
    filename: 'sample_cardiomegaly.png',
    title: 'Cardiomegaly',
    description: 'Enlarged cardiac silhouette with cardiothoracic ratio exceeding 50%.',
    category: 'Cardiomegaly'
  },
  {
    key: 'pneumonia',
    filename: 'sample_pneumonia.png',
    title: 'Pneumonia Finding',
    description: 'Consolidation and alveolar infiltrates across pulmonary lobes.',
    category: 'Pneumonia'
  },
  {
    key: 'effusion',
    filename: 'sample_effusion.png',
    title: 'Pleural Effusion',
    description: 'Blunting of costophrenic sulcus indicating fluid accumulation.',
    category: 'Effusion'
  },
  {
    key: 'multidisease',
    filename: 'sample_multidisease.png',
    title: 'Multi-Pathology',
    description: 'Co-occurrence of infiltration, effusion, and atelectasis.',
    category: 'Complex'
  }
];

interface SamplePickerProps {
  onSelectSample: (sample: SampleItem) => void;
  selectedFilename?: string;
}

export default function SampleXrayPicker({ onSelectSample, selectedFilename }: SamplePickerProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-sky-600" />
          <h3 className="text-sm font-bold text-slate-900">
            Or Test With Verified Clinical Samples
          </h3>
        </div>
        <span className="text-xs text-slate-500 font-medium">Click to load instantly</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {SAMPLE_XRAYS.map((sample) => {
          const isSelected = selectedFilename === sample.filename;
          return (
            <button
              key={sample.key}
              type="button"
              onClick={() => onSelectSample(sample)}
              className={`group relative text-left rounded-xl border p-2.5 transition-all ${
                isSelected
                  ? 'border-sky-500 bg-sky-50/50 ring-2 ring-sky-500/20'
                  : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/70'
              }`}
            >
              {isSelected && (
                <div className="absolute top-2 right-2 z-10 rounded-full bg-sky-600 text-white p-0.5 shadow">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                </div>
              )}
              <div className="aspect-square w-full rounded-lg overflow-hidden bg-slate-900 mb-2 relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`/samples/${sample.filename}`}
                  alt={sample.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                />
              </div>
              <div className="text-xs font-bold text-slate-800 truncate mb-0.5">
                {sample.title}
              </div>
              <div className="text-[10px] text-slate-500 line-clamp-2 leading-tight">
                {sample.description}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
