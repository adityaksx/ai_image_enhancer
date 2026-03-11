import os
import json
import shutil
import numpy as np
import cv2

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from routes.image_routes import router as image_router
from config import INPUT_DIR, OUTPUT_DIR, MODELS_DIR, WEB_DIR

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Image Enhancer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("✅ Folders ready: input/ output/ models/")

# ─── Routes ───────────────────────────────────────────────────────────────────

app.include_router(image_router, prefix="/api")

# ─── Serve Output Images ──────────────────────────────────────────────────────

@app.get("/output-img/{filename}")
def serve_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found in output/")

# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "input_dir":  os.path.exists(INPUT_DIR),
        "output_dir": os.path.exists(OUTPUT_DIR),
        "models_dir": os.path.exists(MODELS_DIR),
    }

# ─── Mount Static ─────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
