import logging
import sys
import json
import re
import contextvars
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from app.core.config import settings

# Context Variable for Trace ID
trace_id_ctx = contextvars.ContextVar("trace_id", default=None)


class HealthCheckFilter(logging.Filter):
    """Filters out health check endpoint logs to reduce noise."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for path in settings.LOG_EXCLUDED_PATHS:
            if f"GET {path} " in msg:  # Simple match for Uvicorn access logs
                return False
        return True


class PIISanitizationFilter(logging.Filter):
    """
    Redacts sensitive PII patterns from log messages.
    
    Patterns redacted:
        - Aadhaar numbers (12 digits with optional spaces)
        - PAN numbers (XXXXX0000X format)
        - Voter ID numbers (3 letters + 7 digits)
        - Driving License numbers (state code + digits)
        - Passport numbers (letter + 7 digits)
    """
    
    PII_PATTERNS = [
        # Aadhaar: 12 digits with optional spaces (e.g., "1234 5678 9012" or "123456789012")
        (re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'), '[AADHAAR-REDACTED]'),
        # PAN: 5 letters + 4 digits + 1 letter (e.g., "ABCDE1234F")
        (re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'), '[PAN-REDACTED]'),
        # Voter ID: 3 letters + 7 digits (e.g., "ABC1234567")
        (re.compile(r'\b[A-Z]{3}\d{7}\b'), '[VOTERID-REDACTED]'),
        # DL Number: 2 letters + 2 digits + space + 11 digits (e.g., "MH12 20190012345")
        (re.compile(r'\b[A-Z]{2}\d{2}\s?\d{11}\b'), '[DL-REDACTED]'),
        # Passport: Letter + 7 digits (e.g., "A1234567")
        (re.compile(r'\b[A-Z]\d{7}\b'), '[PASSPORT-REDACTED]'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Apply PII redaction patterns to log message."""
        if record.msg:
            message = str(record.msg)
            for pattern, replacement in self.PII_PATTERNS:
                message = pattern.sub(replacement, message)
            record.msg = message
        
        # Also sanitize args if present
        if record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.PII_PATTERNS:
                        arg = pattern.sub(replacement, arg)
                sanitized_args.append(arg)
            record.args = tuple(sanitized_args)
        
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
    
    # Apply PII sanitization filter to all handlers
    pii_filter = PIISanitizationFilter()
    logger.addFilter(pii_filter)
    file_handler.addFilter(pii_filter)
    
    # Ensure uvicorn loggers propagate or use our handlers
    logging.getLogger("uvicorn.access").handlers = [stream_handler, file_handler]
    logging.getLogger("uvicorn.error").handlers = [stream_handler, file_handler]
