import os
import json
import shutil
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from services.pipeline import build_and_run_pipeline
from services.folder_processor import process_folder, get_folder_preview
from services.analyzer import get_image_stats
from config import INPUT_DIR, OUTPUT_DIR, SUPPORTED_EXTENSIONS

router = APIRouter()


# ─── Process Single Image ─────────────────────────────────────────────────────

@router.post("/process-image")
async def process_image(
    file:    UploadFile = File(...),
    options: str        = Form(...),
    params:  str        = Form(default="{}")
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {SUPPORTED_EXTENSIONS}"
        )

    os.makedirs(INPUT_DIR, exist_ok=True)
    input_path = os.path.join(INPUT_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    selected_options = [o.strip() for o in options.split(",") if o.strip()]
    if not selected_options:
        raise HTTPException(status_code=400, detail="No enhancement options provided.")

    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        parsed_params = {}

    try:
        output_path = build_and_run_pipeline(input_path, selected_options, parsed_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return JSONResponse({
        "status":  "success",
        "input":   input_path,
        "output":  output_path,
        "options": selected_options,
        "params":  parsed_params
    })


# ─── Process Entire Folder ────────────────────────────────────────────────────

@router.post("/process-folder")
async def process_folder_route(
    options: str = Form(...),
    params:  str = Form(default="{}")
):
    if not os.path.exists(INPUT_DIR):
        raise HTTPException(status_code=400, detail="Input folder does not exist.")

    images = [
        f for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise HTTPException(status_code=400, detail="No supported images found in input/ folder.")

    selected_options = [o.strip() for o in options.split(",") if o.strip()]
    if not selected_options:
        raise HTTPException(status_code=400, detail="No enhancement options provided.")

    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        parsed_params = {}

    results = process_folder(INPUT_DIR, OUTPUT_DIR, selected_options, parsed_params)

    success = [r for r in results if r["status"] == "success"]
    failed  = [r for r in results if r["status"] != "success"]

    return JSONResponse({
        "status":    "done",
        "total":     len(results),
        "success":   len(success),
        "failed":    len(failed),
        "processed": results
    })


# ─── Analyze Image ────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'")

    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    stats = get_image_stats(img)
    return JSONResponse({"status": "analyzed", "stats": stats})


# ─── Get Output Results ───────────────────────────────────────────────────────

@router.get("/results")
def get_results():
    if not os.path.exists(OUTPUT_DIR):
        return JSONResponse({"files": [], "count": 0})

    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    return JSONResponse({"files": files, "count": len(files)})


# ─── Delete All Output Images ─────────────────────────────────────────────────

@router.delete("/results/clear")
def clear_results():
    if not os.path.exists(OUTPUT_DIR):
        return JSONResponse({"status": "nothing to clear", "deleted": 0})

    deleted = 0
    for f in os.listdir(OUTPUT_DIR):
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
            os.remove(os.path.join(OUTPUT_DIR, f))
            deleted += 1

    return JSONResponse({"status": "cleared", "deleted": deleted})


# ─── Folder Preview ───────────────────────────────────────────────────────────

@router.get("/folder-preview")
def folder_preview():
    data = get_folder_preview()
    return JSONResponse(data)
