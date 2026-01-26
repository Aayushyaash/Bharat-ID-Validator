import logging
import sys
import json
import contextvars
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from app.core.config import settings

# Context Variable for Trace ID
trace_id_ctx = contextvars.ContextVar("trace_id", default=None)

class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for path in settings.LOG_EXCLUDED_PATHS:
            if f"GET {path} " in msg: # Simple match for Uvicorn access logs
                return False
        return True

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Get trace_id from context var
        trace_id = trace_id_ctx.get()
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "trace_id": trace_id
        }
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging():
    """
    Configures logging to output to console and a rotating log file.
    """
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "app.log"

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    json_formatter = JSONFormatter()

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(stream_handler)

    # File Handler (Rotating - Daily, keep 7 days)
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    # Apply HealthCheckFilter to Uvicorn access logger
    health_filter = HealthCheckFilter()
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(health_filter)
    
    # Ensure uvicorn loggers propagate or use our handlers
    logging.getLogger("uvicorn.access").handlers = [stream_handler, file_handler]
    logging.getLogger("uvicorn.error").handlers = [stream_handler, file_handler]
