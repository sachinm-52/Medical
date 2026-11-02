-- MedVision AI Database Schema (PostgreSQL / Supabase Compatible)

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Models Registry Table
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) UNIQUE NOT NULL,
    architecture VARCHAR(100) NOT NULL, -- 'DenseNet121', 'ResNet50', 'EfficientNet-B0'
    dataset_name VARCHAR(100) NOT NULL, -- 'MedMNIST (ChestMNIST)', 'Chest X-ray Pneumonia Dataset', 'NIH ChestX-ray14', 'CheXpert'
    model_version VARCHAR(50) DEFAULT '1.0.0',
    weights_path VARCHAR(500),
    status VARCHAR(50) DEFAULT 'ready', -- 'ready', 'training', 'not_trained'
    task_type VARCHAR(50) DEFAULT 'multi_label', -- 'binary', 'multi_label'
    num_classes INTEGER DEFAULT 14,
    class_labels JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Analyses Log Table
CREATE TABLE IF NOT EXISTS analyses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    image_filename VARCHAR(255) NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    gradcam_path VARCHAR(500),
    prediction_results JSONB NOT NULL,
    processing_time DOUBLE PRECISION NOT NULL,
    status VARCHAR(50) DEFAULT 'completed',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Model Evaluation Metrics Table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL, -- 'roc_auc', 'pr_auc', 'accuracy', 'f1_score', 'per_class', 'training_history'
    metric_value DOUBLE PRECISION,
    metric_data JSONB,
    evaluation_split VARCHAR(50) DEFAULT 'test',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_analyses_model_id ON analyses(model_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_id ON model_metrics(model_id);
