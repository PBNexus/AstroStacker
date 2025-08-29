import json
from datetime import datetime
from pathlib import Path
from config import LOGS_DIR

class FrontendLogBuffer:
    def __init__(self, session_id):
        self.session_id = session_id
        self.logs = []
        self.log_file = LOGS_DIR / f"session_{session_id}.json"
    
    def add_log(self, message, level="info"):
        """Add a log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(entry)
        self._save_to_file()
    
    def get_logs(self):
        """Get all logs for frontend"""
        return self.logs
    
    def _save_to_file(self):
        """Save logs to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Error saving logs: {e}")