'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  UploadCloud,
  History,
  BarChart3,
  Database,
  ShieldAlert,
  FileText
} from 'lucide-react';

const NAV_ITEMS = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Upload & Analyze', href: '/upload', icon: UploadCloud },
  { name: 'Prediction History', href: '/history', icon: History },
  { name: 'Model Performance', href: '/models', icon: BarChart3 },
  { name: 'Datasets Provenance', href: '/datasets', icon: Database },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-slate-200 bg-white flex flex-col justify-between shrink-0 min-h-[calc(100vh-4rem)]">
      <div className="p-4 space-y-6">
        <div>
          <p className="px-3 text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
            Navigation
          </p>
          <nav className="space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-sky-50 text-sky-700 shadow-sm border border-sky-100 font-semibold'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 ${
                      isActive ? 'text-sky-600' : 'text-slate-500'
                    }`}
                  />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Explainability Callout */}
        <div className="rounded-xl border border-sky-100 bg-gradient-to-b from-sky-50/70 to-white p-3.5 shadow-sm">
          <div className="flex items-center gap-2 text-sky-800 font-semibold text-xs mb-1">
            <span className="flex h-2 w-2 rounded-full bg-sky-500"></span>
            Grad-CAM Interpretable AI
          </div>
          <p className="text-[11px] leading-relaxed text-slate-600">
            Visual saliency heatmaps uncover radiographic anatomical features driving multi-disease classification.
          </p>
        </div>
      </div>

      {/* Medical Safety Disclaimer Notice */}
      <div className="p-4 border-t border-slate-200 bg-slate-50/60">
        <div className="flex items-start gap-2.5 text-amber-800">
          <ShieldAlert className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-[11px] leading-tight text-slate-600">
            <strong className="text-amber-800 block mb-0.5">Research Use Only</strong>
            Model outputs are not intended as clinical diagnostic advice or treatment planning.
          </div>
        </div>
      </div>
    </aside>
  );
}
