"""
MedVision AI Backend Main Application.
FastAPI REST API Server for Chest X-ray Classification and Grad-CAM Explainability.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database.session import engine, Base, SessionLocal
from backend.app.services.model_service import model_service
from backend.app.routes.api_routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables and register initial model records
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        model_service.sync_models(db)
    finally:
        db.close()
    yield
    # Shutdown

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Explainable Multi-Disease Detection Using Deep Learning and Grad-CAM on Chest X-rays",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories for uploaded images, gradcam output, and sample X-rays
app.mount("/api/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")
app.mount("/api/storage", StaticFiles(directory=str(settings.STORAGE_DIR)), name="storage")
app.mount("/api/samples", StaticFiles(directory=str(settings.SAMPLE_DIR)), name="samples")

# Include API endpoints
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
def root():
    return {
        "project": "MedVision AI",
        "description": "Explainable Multi-Disease Detection Using Chest X-rays",
        "version": "1.0.0",
        "docs_url": "/docs",
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
