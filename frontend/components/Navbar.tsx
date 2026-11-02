'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { Activity, ShieldCheck, Cpu, HardDrive } from 'lucide-react';
import { checkBackendHealth } from '../services/api';

export default function Navbar() {
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    checkBackendHealth()
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: 'offline', device: 'N/A' }));
  }, []);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white/95 backdrop-blur-md">
      <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-600 to-teal-500 text-white shadow-md shadow-sky-500/20 group-hover:scale-105 transition-transform">
              <Activity className="h-6 w-6" />
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight text-slate-900 flex items-center gap-1.5">
                MedVision <span className="text-sky-600 font-extrabold">AI</span>
              </span>
              <span className="text-[10px] uppercase tracking-wider text-slate-600 block -mt-1 font-semibold">
                Chest Radiography Intelligence
              </span>
            </div>
          </Link>
        </div>

        {/* System & Hardware Health Status Badges */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-xs text-slate-600">
            <Cpu className="h-3.5 w-3.5 text-slate-500" />
            <span>Device: <strong className="text-slate-800">{health?.device || 'Detecting...'}</strong></span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-xs">
            <span
              className={`h-2 w-2 rounded-full ${
                health?.status === 'operational'
                  ? 'bg-emerald-500 animate-pulse'
                  : 'bg-amber-500'
              }`}
            />
            <span className="text-slate-600 font-medium">
              {health?.status === 'operational' ? 'API Online' : 'Connecting...'}
            </span>
          </div>

          <Link
            href="/upload"
            className="hidden md:inline-flex items-center justify-center rounded-lg bg-sky-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-sky-700 transition-colors"
          >
            New Analysis
          </Link>
        </div>
      </div>
    </header>
  );
}
