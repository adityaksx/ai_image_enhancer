import os

# ─── Directories ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(BASE_DIR, "input")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
WEB_DIR     = os.path.join(BASE_DIR, "web")

# ─── Model Paths ──────────────────────────────────────────────────────────────
GFPGAN_MODEL_PATH    = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")
REALESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
SWINIR_MODEL_PATH    = os.path.join(MODELS_DIR, "SwinIR_denoising.pth")

# ─── Model Settings ───────────────────────────────────────────────────────────
REALESRGAN_SCALE     = 4      # internal model scale (do not change)
REALESRGAN_OUTSCALE  = 2      # 2x safe for 6GB VRAM, 4x may OOM on large images
REALESRGAN_TILE      = 256    # 256 = safe for RTX 3050 6GB
REALESRGAN_HALF      = True   # fp16 ON — 2x faster on Ampere (RTX 3050)

GFPGAN_UPSCALE       = 1      # keep at 1, upscaling handled by RealESRGAN separately
GFPGAN_ARCH          = "clean"
GFPGAN_CHANNEL_MULT  = 2

SWINIR_IMG_SIZE      = 128
SWINIR_WINDOW_SIZE   = 8
SWINIR_NOISE_LEVEL   = 15     # 15 = light denoise, 25 = medium, 50 = heavy

# ─── Processing ───────────────────────────────────────────────────────────────
OUTPUT_JPEG_QUALITY  = 95
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Auto-Analyzer Thresholds ─────────────────────────────────────────────────
AUTO_BRIGHTNESS_THRESHOLD = 85    # below this → add lighting fix
AUTO_NOISE_THRESHOLD      = 12    # above this → add denoise
AUTO_BLUR_THRESHOLD       = 120   # below this → add sharpen

# ─── Slider Param Defaults (fallback if frontend sends nothing) ───────────────
DEFAULT_LIGHTING_CLIP    = 3.0    # CLAHE clip limit (1.0 – 6.0)
DEFAULT_COLOR_SAT        = 1.25   # saturation multiplier (1.0 – 2.0)
DEFAULT_DENOISE_LEVEL    = 15     # SwinIR noise level (5 / 15 / 25 / 50)
DEFAULT_SHARPEN_STRENGTH = 1.5    # unsharp mask weight (1.0 – 3.0)
DEFAULT_UPSCALE_FACTOR   = 2      # output scale factor (1 – 4)
