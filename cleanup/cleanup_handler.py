import signal
import shutil
import sys
from pathlib import Path
from config import TEMP_DIR
from logger.backend_logger import backend_logger

class CleanupHandler:
    def __init__(self):
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        # Register cleanup_and_exit to be called on SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, self.cleanup_and_exit)
        signal.signal(signal.SIGTERM, self.cleanup_and_exit)
    
    def cleanup_and_exit(self, signum, frame):
        """
        Performs cleanup of temporary files when a shutdown signal is received.
        This method is called by signal handlers.
        It does NOT call sys.exit() directly, allowing the application to shut down gracefully.
        """
        backend_logger.info("Initiating graceful shutdown and temporary file cleanup...")
        self.cleanup_all_temp_dirs()
        backend_logger.info("Temporary files cleaned. Allowing application to exit.")
        # Removed sys.exit(0) to prevent SystemExit traceback during uvicorn shutdown.
        # Uvicorn will handle the process exit after this function completes.

    def cleanup_all_temp_dirs(self):
        """Cleans up all session temporary directories."""
        print("\n🧹 Cleaning up all temporary files...")
        try:
            if TEMP_DIR.exists():
                # Remove all contents within TEMP_DIR but keep TEMP_DIR itself
                for item in TEMP_DIR.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink() # Remove files directly in TEMP_DIR
                backend_logger.info("All temporary session directories cleaned.")
                print("Temporary files cleaned")
            else:
                backend_logger.info("Temporary directory does not exist, no cleanup needed.")
        except Exception as e:
            backend_logger.error(f"Error during full cleanup: {e}")
            print(f"Error during cleanup: {e}")
        
        backend_logger.info("Graceful shutdown completed")
        print("Goodbye, stargazer!")
    
    def cleanup_session(self, session_id: str):
        """Clean up specific session folder"""
        session_dir = TEMP_DIR / f"session_{session_id}"
        if session_dir.exists():
            shutil.rmtree(session_dir)
            backend_logger.info(f"Cleaned up session {session_id}")
        else:
            backend_logger.warning(f"Session directory {session_dir} not found for cleanup.")

# Initialize cleanup handler
cleanup_handler = CleanupHandler()

