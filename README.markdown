# AstroStacker (local copy)

✨ Quick snapshot
AstroStacker is a local FastAPI-based image stacking pipeline aimed at astrophotography processing: calibration, hot-pixel removal, alignment, background subtraction and final stacking — with a tiny web UI for uploads and previews.

> This README was auto-generated for the project contents you provided. Feel free to edit it to match your voice! 😎🚀

---

## Highlights
- FastAPI backend with a web UI (see `templates/` + `static/`).
- Modular code organized under the `alignment/`, `calibration/`, `background/`, `cleanup/` and `validation/` packages.
- Configurable runtime directories in `config.py` (data, temp, output, logs).
- Example entrypoint: `main.py` (runs via `uvicorn` when executed directly).

---

## Quickstart — run locally (5–10 mins)
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows (PowerShell)

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt

3. Start the server
    ```bash
    python main.py
    OR
    uvicorn main:app