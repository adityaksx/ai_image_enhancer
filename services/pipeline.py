import cv2
import os
import numpy as np
from services.enhancements import (
    fix_lighting,
    enhance_colors,
    denoise,
    sharpen,
    face_enhance,
    upscale
)
from services.analyzer import analyze_image
from config import (
    OUTPUT_DIR, OUTPUT_JPEG_QUALITY,
    DEFAULT_LIGHTING_CLIP,
    DEFAULT_COLOR_SAT,
    DEFAULT_DENOISE_LEVEL,
    DEFAULT_SHARPEN_STRENGTH,
    DEFAULT_UPSCALE_FACTOR
)

# ─── Step Order ───────────────────────────────────────────────────────────────
STEP_ORDER = ["lighting", "color", "denoise", "face", "sharpen", "upscale"]


# ─── Pipeline Builder ─────────────────────────────────────────────────────────

def _build_map(params: dict) -> dict:
    """
    Builds the pipeline function map with slider param values.
    Falls back to config defaults if param not provided.
    """
    return {
        "lighting": lambda img: fix_lighting(img,
                        clip_limit=params.get("lighting_clip",    DEFAULT_LIGHTING_CLIP)),
        "color":    lambda img: enhance_colors(img,
                        sat_factor=params.get("color_sat",        DEFAULT_COLOR_SAT)),
        "denoise":  lambda img: denoise(img),
        "face":     lambda img: face_enhance(img),
        "sharpen":  lambda img: sharpen(img,
                        strength=params.get("sharpen_strength",   DEFAULT_SHARPEN_STRENGTH)),
        "upscale":  lambda img: upscale(img),
    }


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def build_and_run_pipeline(image_path: str, options: list, params: dict = None) -> str:  # ← None not {}
    """
    Loads an image, builds a pipeline from selected options + slider params,
    runs each step in correct order, saves and returns output path.
    """
    if params is None:
        params = {}

    # ── Load Image ────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_h, original_w = img.shape[:2]
    print(f"\n  [pipeline] Image:      {os.path.basename(image_path)}")
    print(f"  [pipeline] Resolution: {original_w}x{original_h}")
    print(f"  [pipeline] Options:    {options}")
    print(f"  [pipeline] Params:     {params}")

    # ── Resolve Auto Steps ────────────────────────────────────────────────────
    if "auto" in options:
        print("  [pipeline] Auto-detect enabled → running analyzer...")
        auto_steps     = analyze_image(img)
        manual_options = [o for o in options if o != "auto"]
        merged_options = list(dict.fromkeys(auto_steps + manual_options))
        print(f"  [pipeline] Merged options: {merged_options}")
    else:
        merged_options = options

    # ── Build Ordered Pipeline ────────────────────────────────────────────────
    ordered_steps = [s for s in STEP_ORDER if s in merged_options]

    if not ordered_steps:
        print("  [pipeline] No valid steps — saving original image as-is")

    print(f"  [pipeline] Final pipeline: {ordered_steps}")

    # ── Build Function Map With Params ────────────────────────────────────────
    pipeline_map = _build_map(params)

    # ── Run Each Step ─────────────────────────────────────────────────────────
    for step in ordered_steps:
        print(f"  [pipeline] → Running: {step}")
        try:
            img = pipeline_map[step](img)
        except Exception as e:
            print(f"  [pipeline] ✗ Step '{step}' failed: {e} — skipping")
            continue

    # ── Save Output ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base     = os.path.splitext(os.path.basename(image_path))[0]
    out_name = f"{base}_enhanced.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY])

    final_h, final_w = img.shape[:2]
    print(f"  [pipeline] ✓ Saved:      {out_path}")
    print(f"  [pipeline] Final res:    {final_w}x{final_h}\n")

    return out_path


# ─── Preview Pipeline (no save) ───────────────────────────────────────────────

def preview_pipeline(image_path: str, options: list, params: dict = None) -> np.ndarray:  # ← None not {}
    """
    Runs the pipeline but returns image array instead of saving.
    """
    if params is None:
        params = {}

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if "auto" in options:
        auto_steps     = analyze_image(img)
        manual_options = [o for o in options if o != "auto"]
        merged_options = list(dict.fromkeys(auto_steps + manual_options))
    else:
        merged_options = options

    ordered_steps = [s for s in STEP_ORDER if s in merged_options]
    pipeline_map  = _build_map(params)

    for step in ordered_steps:
        try:
            img = pipeline_map[step](img)
        except Exception as e:
            print(f"  [preview] Step '{step}' failed: {e} — skipping")
            continue

    return img
