import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from stacker import AstroStacker
from config import TEMP_DIR, OUTPUT_DIR, SUPPORTED_INPUT_FORMATS, MAX_FILE_SIZE
from logger.backend_logger import backend_logger
from cleanup.cleanup_handler import cleanup_handler
from file_loader import FileLoader # Ensure FileLoader is imported
from validation.input_validator import InputValidator

router = APIRouter()

def normalize_optional_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    cleaned = [v for v in values if v.strip()]
    return cleaned if cleaned else None

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...)
):
    """Upload files for stacking"""
    try:
        session_dir = TEMP_DIR / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)
        
        uploaded_files_info = []
        
        for file_obj in files:
            if file_obj.size > MAX_FILE_SIZE:
                raise HTTPException(400, f"File {file_obj.filename} too large (max {MAX_FILE_SIZE / (1024*1024)}MB)")
            
            file_ext = Path(file_obj.filename).suffix.lower()
            if file_ext not in SUPPORTED_INPUT_FORMATS:
                raise HTTPException(400, f"Unsupported file format: {file_obj.filename}. Supported formats: {', '.join(SUPPORTED_INPUT_FORMATS)}")
            
            file_type_prefix = ""
            if file_obj.filename.startswith("light_"):
                file_type_prefix = "light_"
            elif file_obj.filename.startswith("dark_"):
                file_type_prefix = "dark_"
            elif file_obj.filename.startswith("bias_"):
                file_type_prefix = "bias_"
            elif file_obj.filename.startswith("flat_"):
                file_type_prefix = "flat_"
            
            filename_cleaned = file_obj.filename
            if file_type_prefix and filename_cleaned.startswith(file_type_prefix):
                filename_cleaned = filename_cleaned[len(file_type_prefix):]

            unique_filename = f"{file_type_prefix}{filename_cleaned}"
            file_path = session_dir / unique_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_obj.file, buffer)
            
            uploaded_files_info.append({"filename": unique_filename, "size": file_obj.size})
            backend_logger.info(f"Uploaded {unique_filename} to {session_dir}")

        return JSONResponse({
            "status": "success",
            "message": "Files uploaded successfully",
            "files": uploaded_files_info
        })
    except HTTPException as e:
        backend_logger.error(f"File upload error for session {session_id}: {e.detail}")
        raise e
    except Exception as e:
        backend_logger.error(f"Unexpected error during file upload for session {session_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error during upload: {e}")

@router.post("/validate")
async def validate_metadata_endpoint(request: Request):
    try:
        form = await request.form()
        
        session_id = form.get('session_id')
        if not session_id:
            raise HTTPException(422, "Missing session_id")

        def extract_clean_list(field):
            raw = form.getlist(field)
            return [f for f in raw if f.strip()]

        light_files = extract_clean_list('light_files')
        dark_files = extract_clean_list('dark_files')
        bias_files = extract_clean_list('bias_files')
        flat_files = extract_clean_list('flat_files')

        if not light_files:
            raise HTTPException(422, "At least one light frame is required.")

        session_dir = TEMP_DIR / f"session_{session_id}"
        if not session_dir.exists():
            raise HTTPException(404, "Session directory not found.")

        # Create an instance of FileLoader
        file_loader_instance = FileLoader() # ADDED: Instantiate FileLoader

        light_file_paths = [session_dir / f for f in light_files]
        dark_file_paths = [session_dir / f for f in dark_files] if dark_files else None
        bias_file_paths = [session_dir / f for f in bias_files] if bias_files else None
        flat_file_paths = [session_dir / f for f in flat_files] if flat_files else None

        # Call extract_metadata on the instance
        light_frame_metadata = [file_loader_instance.extract_metadata(p) for p in light_file_paths if p.exists()] # MODIFIED
        dark_frame_metadata = [file_loader_instance.extract_metadata(p) for p in dark_file_paths] if dark_file_paths else [] # MODIFIED

        warnings = InputValidator.validate_metadata(light_frame_metadata, dark_frame_metadata)

        return JSONResponse({
            "status": "success",
            "message": "Metadata validation performed.",
            "validation_warnings": warnings
        })
    except HTTPException as e:
        log_session_id = session_id if 'session_id' in locals() else 'unknown_session'
        backend_logger.error(f"Metadata validation error for session {log_session_id}: {e.detail}")
        raise e
    except Exception as e:
        log_session_id = session_id if 'session_id' in locals() else 'unknown_session'
        backend_logger.error(f"Unexpected error during metadata validation for session {log_session_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error during metadata validation: {e}")


@router.post("/stack")
async def stack_images_endpoint(request: Request):
    """
    Initiates the image stacking process.
    """
    try:
        form = await request.form()

        session_id = form.get("session_id")
        if not session_id:
            raise HTTPException(422, "Missing session_id")

        def clean(field):
            raw = form.getlist(field)
            return [f for f in raw if f.strip()]

        light_files = clean("light_files")
        dark_files = clean("dark_files")
        bias_files = clean("bias_files")
        flat_files = clean("flat_files")
        output_format = form.get("output_format", "png")
        background_method = form.get("background_subtraction_method", "median")

        if not light_files:
            raise HTTPException(422, "No light frames provided.")

        session_dir = TEMP_DIR / f"session_{session_id}"
        if not session_dir.exists():
            raise HTTPException(404, "Session not found.")

        light_paths = [session_dir / f for f in light_files]
        dark_paths = [session_dir / f for f in dark_files] if dark_files else None
        bias_paths = [session_dir / f for f in bias_files] if bias_files else None
        flat_paths = [session_dir / f for f in flat_files] if flat_files else None

        camera_model = None
        # FileLoader instance is created within AstroStacker for consistency
        # For metadata extraction here, we can create a temporary instance or rely on AstroStacker's
        # Let's create a temporary instance for metadata extraction if needed outside AstroStacker
        temp_file_loader = FileLoader() # ADDED: Temporary FileLoader instance for this block
        if light_paths and light_paths[0].exists():
            meta = temp_file_loader.extract_metadata(light_paths[0]) # MODIFIED
            camera_model = meta.get("camera_model")
            if camera_model:
                backend_logger.info(f"Camera model detected for stacking: {camera_model}")
            else:
                backend_logger.info("Camera model not detected for stacking.")

        stacker = AstroStacker(session_id)
        output_file, preview_file = stacker.stack_images(
            light_file_paths=light_paths,
            dark_file_paths=dark_paths,
            bias_file_paths=bias_paths,
            flat_file_paths=flat_paths,
            output_format=output_format,
            background_subtraction_method=background_method,
            camera_model=camera_model
        )

        download_url = f"/output/session_{session_id}/{output_file}"
        preview_url = f"/output/session_{session_id}/{preview_file}"

        return JSONResponse({
            "status": "success",
            "message": "Image stacking completed successfully!",
            "download_url": download_url,
            "preview_url": preview_url
        })

    except HTTPException as e:
        log_session_id = session_id if 'session_id' in locals() else 'unknown_session'
        backend_logger.error(f"Stacking error for session {log_session_id}: {e.detail}")
        raise e
    except ValueError as e:
        log_session_id = session_id if 'session_id' in locals() else 'unknown_session'
        backend_logger.error(f"Stacking validation error for session {log_session_id}: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        log_session_id = session_id if 'session_id' in locals() else 'unknown_session'
        backend_logger.error(f"Unexpected error during image stacking for session {log_session_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error during stacking: {e}")

@router.get("/logs/{session_id}")
async def get_logs(session_id: str):
    """Retrieve logs for a specific session"""
    from logger.frontend_logger import FrontendLogBuffer
    try:
        log_buffer = FrontendLogBuffer(session_id)
        logs = log_buffer.get_logs()
        return JSONResponse({"logs": logs})
    except Exception as e:
        backend_logger.error(f"Error retrieving logs for session {session_id}: {e}")
        raise HTTPException(500, f"Error retrieving logs: {e}")

@router.post("/cleanup_session/{session_id}")
async def cleanup_single_session(session_id: str):
    """Clean up a specific session's temporary files."""
    try:
        cleanup_handler.cleanup_session(session_id)
        return JSONResponse({"status": "success", "message": f"Session {session_id} cleaned up."})
    except Exception as e:
        backend_logger.error(f"Error cleaning up session {session_id}: {e}")
        raise HTTPException(500, f"Error cleaning up session: {e}")
