import cv2
import numpy as np
import torch
import os
import sys
from config import (
    MODELS_DIR,
    GFPGAN_MODEL_PATH,
    REALESRGAN_MODEL_URL,
    REALESRGAN_SCALE,
    REALESRGAN_OUTSCALE,
    REALESRGAN_TILE,
    REALESRGAN_HALF,
    GFPGAN_UPSCALE,
    GFPGAN_ARCH,
    GFPGAN_CHANNEL_MULT,
    SWINIR_MODEL_PATH,
    SWINIR_IMG_SIZE,
    SWINIR_WINDOW_SIZE,
    SWINIR_NOISE_LEVEL,
    DEFAULT_LIGHTING_CLIP,
    DEFAULT_COLOR_SAT,
    DEFAULT_SHARPEN_STRENGTH,
)


# ─── OpenCV — Lighting Fix ────────────────────────────────────────────────────

def fix_lighting(img: np.ndarray, clip_limit: float = DEFAULT_LIGHTING_CLIP) -> np.ndarray:
    """CLAHE on L channel of LAB — fixes dark/uneven lighting without blowing out highlights."""
    lab        = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    merged     = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ─── OpenCV — Color Enhancement ───────────────────────────────────────────────

def enhance_colors(img: np.ndarray, sat_factor: float = DEFAULT_COLOR_SAT) -> np.ndarray:
    """Boosts saturation and slightly raises brightness via HSV."""
    hsv          = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor,        0, 255)  # saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05,              0, 255)  # brightness +5%
    return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)


# ─── OpenCV — Sharpen ─────────────────────────────────────────────────────────

def sharpen(img: np.ndarray, strength: float = DEFAULT_SHARPEN_STRENGTH) -> np.ndarray:
    """Unsharp mask sharpening — crisp without harsh edges."""
    blurred   = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, strength, blurred, -(strength - 1), 0)
    return sharpened


# ─── SwinIR — Denoise ─────────────────────────────────────────────────────────

def denoise(img: np.ndarray) -> np.ndarray:
    """
    Uses SwinIR for AI-based color denoising.
    Falls back to OpenCV fastNlMeans if model not found or error occurs.
    """
    try:
        swinir_code = os.path.join(MODELS_DIR, "SwinIR")
        if swinir_code not in sys.path:
            sys.path.insert(0, swinir_code)

        from models.network_swinir import SwinIR as SwinIRModel

        if not os.path.exists(SWINIR_MODEL_PATH):
            print("  [denoise] SwinIR weights not found → OpenCV fallback")
            return _denoise_opencv(img)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [denoise] SwinIR running on {device}")

        model = SwinIRModel(
            upscale=1,
            in_chans=3,
            img_size=SWINIR_IMG_SIZE,
            window_size=SWINIR_WINDOW_SIZE,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv"
        ).to(device)

        pretrained = torch.load(SWINIR_MODEL_PATH, map_location=device)
        param_key  = "params" if "params" in pretrained else None
        model.load_state_dict(
            pretrained[param_key] if param_key else pretrained,
            strict=False
        )
        model.eval()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor  = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad to multiple of window_size
        _, _, h, w = tensor.shape
        pad_h = (SWINIR_WINDOW_SIZE - h % SWINIR_WINDOW_SIZE) % SWINIR_WINDOW_SIZE
        pad_w = (SWINIR_WINDOW_SIZE - w % SWINIR_WINDOW_SIZE) % SWINIR_WINDOW_SIZE
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            out = model(tensor)

        out = out[:, :, :h, :w]
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"  [denoise] SwinIR error: {e} → OpenCV fallback")
        return _denoise_opencv(img)


def _denoise_opencv(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


# ─── GFPGAN — Face Enhancement ────────────────────────────────────────────────

def face_enhance(img: np.ndarray) -> np.ndarray:
    """
    Uses GFPGAN v1.4 for face restoration.
    Skips gracefully if no face is found or model errors.
    """
    try:
        from gfpgan import GFPGANer

        restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=GFPGAN_UPSCALE,
            arch=GFPGAN_ARCH,
            channel_multiplier=GFPGAN_CHANNEL_MULT,
            bg_upsampler=None
        )

        print("  [face_enhance] Running GFPGAN...")
        _, _, output = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        if output is None:
            print("  [face_enhance] No face enhanced, returning original")
            return img

        return output

    except Exception as e:
        print(f"  [face_enhance] GFPGAN error: {e} → returning original")
        return img


# ─── Real-ESRGAN — Upscale ────────────────────────────────────────────────────

def upscale(img: np.ndarray) -> np.ndarray:
    """
    Uses Real-ESRGAN x4plus for image upscaling.
    Tiled mode keeps within 6GB VRAM on RTX 3050.
    Falls back to cv2 Lanczos if model errors.
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=REALESRGAN_SCALE
        )

        upsampler = RealESRGANer(
            scale=REALESRGAN_SCALE,
            model_path=REALESRGAN_MODEL_URL,
            model=model,
            tile=REALESRGAN_TILE,
            tile_pad=10,
            pre_pad=0,
            half=REALESRGAN_HALF
        )

        print(f"  [upscale] Running RealESRGAN {REALESRGAN_OUTSCALE}x...")
        output, _ = upsampler.enhance(img, outscale=REALESRGAN_OUTSCALE)
        return output

    except Exception as e:
        print(f"  [upscale] RealESRGAN error: {e} → cv2 Lanczos fallback")
        return _upscale_opencv(img)


def _upscale_opencv(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
