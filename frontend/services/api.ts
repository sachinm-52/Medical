const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ModelSummary {
  id: number;
  model_name: string;
  architecture: string;
  dataset_name: string;
  model_version: string;
  status: 'ready' | 'not_trained' | 'training';
  task_type: 'binary' | 'multi_label';
  num_classes: number;
  class_labels: string[];
  weights_path?: string;
  created_at: string;
}

export interface DatasetInfo {
  name: string;
  slug: string;
  description: string;
  source_url: string;
  task_type: string;
  num_classes: number;
  classes: string[];
  provenance: string;
  patient_level_split: boolean;
  total_images_note: string;
  license: string;
  uncertainty_handling?: string;
}

export interface DiseasePrediction {
  disease: string;
  score: number;
  threshold: number;
  is_positive: boolean;
  confidence_level: 'High' | 'Moderate' | 'Low' | 'Very Low';
}

export interface PredictionResponse {
  analysis_id: number;
  image_filename: string;
  image_url: string;
  gradcam_url?: string;
  model_id: number;
  model_name: string;
  architecture: string;
  dataset_name: string;
  task_type: string;
  predictions: DiseasePrediction[];
  top_prediction?: DiseasePrediction;
  overall_assessment: string;
  processing_time_ms: number;
  timestamp: string;
  disclaimer: string;
}

export interface GradCamResponse {
  analysis_id: number;
  original_image_url: string;
  gradcam_heatmap_url: string;
  gradcam_overlay_url: string;
  target_disease: string;
  explanation_note: string;
}

export interface HistoryItem {
  id: number;
  model_name: string;
  dataset_name: string;
  image_filename: string;
  image_url: string;
  gradcam_url?: string;
  predictions: DiseasePrediction[];
  top_prediction?: DiseasePrediction;
  overall_assessment: string;
  processing_time: number;
  status: string;
  created_at: string;
}

export interface MetricItem {
  id: number;
  model_id: number;
  model_name: string;
  architecture: string;
  dataset_name: string;
  metric_name: string;
  metric_value?: number;
  metric_data?: any;
  evaluation_split: string;
  created_at: string;
}

export async function checkBackendHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchModels(): Promise<ModelSummary[]> {
  const res = await fetch(`${API_BASE_URL}/api/models`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_BASE_URL}/api/datasets`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function uploadXrayImage(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Image upload failed');
  }
  return res.json();
}

export async function runPredict(
  modelId: number,
  imageFilename: string,
  threshold: number = 0.5
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('model_id', modelId.toString());
  formData.append('image_filename', imageFilename);
  formData.append('threshold', threshold.toString());

  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}

export async function generateGradCam(
  analysisId: number,
  targetClassIdx?: number,
  alpha: number = 0.5
): Promise<GradCamResponse> {
  const res = await fetch(`${API_BASE_URL}/api/gradcam`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      analysis_id: analysisId,
      target_class_idx: targetClassIdx,
      alpha: alpha,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Grad-CAM generation failed');
  }
  return res.json();
}

export async function fetchHistory(
  search?: string,
  modelId?: number
): Promise<{ total: number; items: HistoryItem[] }> {
  const params = new URLSearchParams();
  if (search) params.append('search', search);
  if (modelId) params.append('model_id', modelId.toString());

  const res = await fetch(`${API_BASE_URL}/api/history?${params.toString()}`, {
    cache: 'no-store',
  });
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

export async function fetchHistoryDetail(analysisId: number) {
  const res = await fetch(`${API_BASE_URL}/api/history/${analysisId}`, {
    cache: 'no-store',
  });
  if (!res.ok) throw new Error('Failed to fetch analysis detail');
  return res.json();
}

export async function deleteHistoryItem(analysisId: number) {
  const res = await fetch(`${API_BASE_URL}/api/history/${analysisId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete history item');
  return res.json();
}

export async function fetchMetrics(modelId?: number): Promise<MetricItem[]> {
  const url = modelId
    ? `${API_BASE_URL}/api/metrics?model_id=${modelId}`
    : `${API_BASE_URL}/api/metrics`;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch metrics');
  return res.json();
}

export function getPdfReportUrl(analysisId: number) {
  return `${API_BASE_URL}/api/reports/${analysisId}/pdf`;
}

export function getFullImageUrl(relUrl: string) {
  if (!relUrl) return '';
  if (relUrl.startsWith('http://') || relUrl.startsWith('https://')) return relUrl;
  return `${API_BASE_URL}${relUrl}`;
}
