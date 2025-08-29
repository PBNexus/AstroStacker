from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from routes import router
from config import OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR
from logger.backend_logger import backend_logger
from cleanup.cleanup_handler import cleanup_handler
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles app startup and shutdown in FastAPI 2024+"""
    # Startup
    backend_logger.info("AstroStacker v0.4 starting up...")
    backend_logger.info("Ready for astrophotography image stacking")
    yield
    # Shutdown
    backend_logger.info("AstroStacker v0.4 shutting down...")
    cleanup_handler.cleanup_all_temp_dirs()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AstroStacker v0.4",
    description="Professional Astrophotography Image Stacking Tool",
    version="0.4.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include API routes
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    try:
        backend_logger.info("Starting AstroStacker v0.4...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=5500,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        backend_logger.info("Received shutdown signal")
