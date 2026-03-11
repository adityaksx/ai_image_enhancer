import os
import time
from services.pipeline import build_and_run_pipeline
from config import SUPPORTED_EXTENSIONS, INPUT_DIR, OUTPUT_DIR


def process_folder(
    input_dir:  str  = INPUT_DIR,
    output_dir: str  = OUTPUT_DIR,
    options:    list = None,
    params:     dict = None        # ← ADD: slider params from frontend
) -> list:
    """
    Processes all supported images in input_dir through the enhancement pipeline.
    Returns a list of result dicts with status for each image.
    """
    if options is None:
        options = ["auto"]

    if params is None:
        params = {}

    os.makedirs(output_dir, exist_ok=True)

    # ─── Collect Images ───────────────────────────────────────────────────────
    images = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        print("  [folder] No supported images found in input folder.")
        return []

    total   = len(images)
    results = []

    print(f"\n  [folder] Found {total} image(s) to process")
    print(f"  [folder] Options: {options}")
    print(f"  [folder] Params:  {params}")
    print(f"  [folder] Output → {output_dir}\n")

    # ─── Process Each Image ───────────────────────────────────────────────────
    for index, filename in enumerate(images, start=1):
        input_path = os.path.join(input_dir, filename)
        print(f"  [{index}/{total}] Processing: {filename}")

        start_time = time.time()

        try:
            output_path = build_and_run_pipeline(input_path, options, params)  # ← pass params
            elapsed     = round(time.time() - start_time, 2)

            print(f"  [{index}/{total}] ✓ Done in {elapsed}s → {output_path}\n")
            results.append({
                "file":   filename,
                "output": output_path,
                "status": "success",
                "time_s": elapsed
            })

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            print(f"  [{index}/{total}] ✗ Failed: {e}\n")
            results.append({
                "file":   filename,
                "output": None,
                "status": f"error: {str(e)}",
                "time_s": elapsed
            })

    # ─── Summary ──────────────────────────────────────────────────────────────
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count  = total - success_count
    total_time    = round(sum(r["time_s"] for r in results), 2)

    print(f"  [folder] ─────────────────────────────────────")
    print(f"  [folder] Done:       {success_count}/{total} succeeded")
    print(f"  [folder] Failed:     {failed_count}")
    print(f"  [folder] Total time: {total_time}s")
    print(f"  [folder] ─────────────────────────────────────\n")

    return results


def get_folder_preview(input_dir: str = INPUT_DIR) -> dict:
    """
    Returns a preview of what's in the input folder before processing.
    Useful for the frontend to show file count before user hits Process Folder.
    """
    if not os.path.exists(input_dir):
        return {"exists": False, "count": 0, "files": []}

    files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    return {
        "exists": True,
        "count":  len(files),
        "files":  sorted(files)
    }
