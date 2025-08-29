# AstroStacker

✨ Quick snapshot
AstroStacker is a local FastAPI-based image stacking pipeline aimed at astrophotography processing: calibration, hot-pixel removal, alignment, background subtraction and final stacking — with a tiny web UI for uploads and previews.

>You will upload lights, darks, biases, and flats (lights is mandatory(ofcourse), rest are optional)
>Uploaded frames can be of the following types: '.jpg', '.jpeg', '.png', '.tiff', '.tif','.fits', '.fit', '.cr2', '.nef', '.dng', '.arw'
>It does the usual processes, images are aligned, calibration frames are made, Hot and Cold Pixels are removed, of the images takes place. and output is given in one of the selected formats- Output files can be of following types: 'fits', 'tiff', 'png', 'jpg'
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