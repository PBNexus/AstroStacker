# AstroStacker

✨ Quick snapshot
AstroStacker is a local FastAPI-based image stacking pipeline aimed at astrophotography processing: calibration, hot-pixel removal, alignment, background subtraction and final stacking — with a tiny web UI for uploads and previews.

Upload the following calibration frames:  
- **Lights** (mandatory)  
- **Darks** (optional)  
- **Biases** (optional)  
- **Flats** (optional)  

Accepted input formats:  
`.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.fits`, `.fit`, `.cr2`, `.nef`, `.dng`, `.arw`  

Processing steps:  
1. Frame alignment  
2. Construction of master calibration frames  
3. Removal of hot and cold pixels  
4. Application of calibration corrections  

Output can be generated in the following formats:  
`fits`, `tiff`, `png`, `jpg`  
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
    ```
    OR
    ```bash
    uvicorn main:app
    ```