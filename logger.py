"""
Advanced logging module for Telegram bot "Golostickery".
Provides centralized logging with rotation, filtering, metrics, and structured output.
"""
import logging
import logging.handlers
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable, Type
from datetime import datetime, timezone
from contextlib import contextmanager
from functools import wraps
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from queue import Queue, Empty
import gzip
import shutil

# Import config or use defaults
try:
    from config import config
    LOGS_DIR = config.paths.logs_dir
    LOG_LEVEL = config.logging.level
    LOG_FORMAT = config.logging.format
    LOG_DATE_FORMAT = config.logging.date_format
    LOG_FILE_MAX_BYTES = config.logging.file_max_bytes
    LOG_FILE_BACKUP_COUNT = config.logging.file_backup_count
    COLORED_OUTPUT = config.logging.colored_output
    LOG_PERFORMANCE_METRICS = config.logging.log_performance_metrics
except ImportError:
    LOGS_DIR = Path("logs")
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
    LOG_FILE_BACKUP_COUNT = 5
    COLORED_OUTPUT = True
    LOG_PERFORMANCE_METRICS = True

# Ensure logs directory exists
LOGS_DIR = Path(LOGS_DIR)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class LogLevel(Enum):
    """Extended log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    METRIC = 25  # Between INFO and WARNING


# Add custom log levels
logging.addLevelName(LogLevel.TRACE.value, "TRACE")
logging.addLevelName(LogLevel.METRIC.value, "METRIC")


@dataclass
class LogContext:
    """Structured context for logging."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    filename: str
    line_number: int
    function_name: str
    thread_name: str
    thread_id: int
    process_id: int
    user_id: Optional[int] = None
    username: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract exception info if present
        error_details = None
        if record.exc_info:
            error_details = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Build context
        context = LogContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            filename=record.filename,
            line_number=record.lineno,
            function_name=record.funcName,
            thread_name=record.threadName,
            thread_id=record.thread,
            process_id=record.process,
            user_id=getattr(record, 'user_id', None),
            username=getattr(record, 'username', None),
            request_id=getattr(record, 'request_id', None),
            duration_ms=getattr(record, 'duration_ms', None),
            error_type=record.exc_info[0].__name__ if record.exc_info else None,
            error_details=error_details,
            extra={k: v for k, v in record.__dict__.items() 
                   if k not in ['name', 'msg', 'args', 'created', 'filename', 
                               'funcName', 'levelname', 'levelno', 'lineno', 
                               'module', 'msecs', 'pathname', 'process', 
                               'processName', 'relativeCreated', 'thread', 
                               'threadName', 'exc_info', 'exc_text', 'stack_info',
                               'user_id', 'username', 'request_id', 'duration_ms']}
        )
        
        return context.to_json()


class ColoredFormatter(logging.Formatter):
    """Enhanced formatter with colors and better formatting."""
    
    # ANSI escape codes
    COLORS = {
        'TRACE': '\033[37m',      # White
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'METRIC': '\033[34m',     # Blue
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors and context."""
        # Save original values
        levelname = record.levelname
        
        # Add color to level name
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname:8}{self.RESET}"
        
        # Add context info if available
        context_parts = []
        if hasattr(record, 'user_id') and record.user_id:
            context_parts.append(f"user:{record.user_id}")
        if hasattr(record, 'username') and record.username:
            context_parts.append(f"@{record.username}")
        if hasattr(record, 'request_id') and record.request_id:
            context_parts.append(f"req:{record.request_id}")
        if hasattr(record, 'duration_ms') and record.duration_ms:
            context_parts.append(f"{record.duration_ms:.1f}ms")
        
        # Add context to message
        if context_parts:
            record.msg = f"[{' '.join(context_parts)}] {record.msg}"
        
        # Format message
        formatted = super().format(record)
        
        # Restore original
        record.levelname = levelname
        
        return formatted


class PerformanceFilter(logging.Filter):
    """Filter that adds performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_times: Dict[int, float] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance data to record."""
        thread_id = threading.get_ident()
        
        # Track request start
        if hasattr(record, 'request_start'):
            self.start_times[thread_id] = time.time()
        
        # Calculate duration
        if hasattr(record, 'request_end') and thread_id in self.start_times:
            duration = (time.time() - self.start_times[thread_id]) * 1000
            record.duration_ms = duration
            del self.start_times[thread_id]
        
        return True


class AsyncRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Asynchronous rotating file handler for better performance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue: Queue = Queue(maxsize=1000)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def emit(self, record: logging.LogRecord):
        """Put record in queue instead of writing directly."""
        try:
            self.queue.put_nowait(record)
        except:
            # If queue is full, fall back to synchronous write
            super().emit(record)
    
    def _worker(self):
        """Background worker to write logs."""
        while True:
            try:
                record = self.queue.get(timeout=1)
                super().emit(record)
            except Empty:
                continue
            except Exception as e:
                print(f"Error in log worker: {e}", file=sys.stderr)
    
    def doRollover(self):
        """Enhanced rollover with compression."""
        super().doRollover()
        
        # Compress old log files
        for i in range(self.backupCount - 1, 0, -1):
            sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
            if Path(sfn).exists() and not sfn.endswith('.gz'):
                self._compress_file(sfn)
    
    def _compress_file(self, filepath: str):
        """Compress log file with gzip."""
        try:
            with open(filepath, 'rb') as f_in:
                with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            Path(filepath).unlink()
        except Exception as e:
            print(f"Failed to compress {filepath}: {e}", file=sys.stderr)


class LoggerManager:
    """Centralized logger management."""
    
    _instance: Optional['LoggerManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'LoggerManager':
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize logger manager."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.request_id_var = threading.local()
        self.metrics_queue: Queue = Queue()
        self.metrics_thread = threading.Thread(target=self._process_metrics, daemon=True)
        self.metrics_thread.start()
    
    def setup_logging(
        self,
        log_level: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = True,
        structured_logs: bool = False,
        colored_output: bool = True,
        async_file_handler: bool = True
    ) -> logging.Logger:
        """Configure logging system."""
        # Determine log level
        level = getattr(logging, (log_level or LOG_LEVEL).upper())
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        if structured_logs:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        
        # Add filters
        perf_filter = PerformanceFilter()
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            if colored_output and sys.stdout.isatty() and not structured_logs:
                console_formatter = ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
                console_handler.setFormatter(console_formatter)
            else:
                console_handler.setFormatter(formatter)
            
            console_handler.addFilter(perf_filter)
            root_logger.addHandler(console_handler)
            self.handlers['console'] = console_handler
        
        # File handlers
        if log_to_file:
            # Main log file
            log_file = LOGS_DIR / "bot.log"
            if async_file_handler:
                file_handler = AsyncRotatingFileHandler(
                    log_file,
                    maxBytes=LOG_FILE_MAX_BYTES,
                    backupCount=LOG_FILE_BACKUP_COUNT,
                    encoding='utf-8'
                )
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=LOG_FILE_MAX_BYTES,
                    backupCount=LOG_FILE_BACKUP_COUNT,
                    encoding='utf-8'
                )
            
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(perf_filter)
            root_logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
            
            # Error log file
            error_file = LOGS_DIR / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            error_handler.addFilter(perf_filter)
            root_logger.addHandler(error_handler)
            self.handlers['error'] = error_handler
            
            # Metrics log file
            if LOG_PERFORMANCE_METRICS:
                metrics_file = LOGS_DIR / "metrics.log"
                metrics_handler = logging.handlers.RotatingFileHandler(
                    metrics_file,
                    maxBytes=LOG_FILE_MAX_BYTES,
                    backupCount=2,
                    encoding='utf-8'
                )
                metrics_handler.setLevel(LogLevel.METRIC.value)
                metrics_handler.setFormatter(StructuredFormatter())
                root_logger.addHandler(metrics_handler)
                self.handlers['metrics'] = metrics_handler
        
        # Configure third-party loggers
        self._configure_third_party_loggers()
        
        # Log startup
        root_logger.info("=" * 80)
        root_logger.info(f"Logging system initialized - {datetime.now()}")
        root_logger.info(f"Log level: {log_level or LOG_LEVEL}")
        root_logger.info(f"Log directory: {LOGS_DIR}")
        root_logger.info(f"Structured logs: {structured_logs}")
        root_logger.info(f"Async file handler: {async_file_handler}")
        root_logger.info("=" * 80)
        
        return root_logger
    
    def _configure_third_party_loggers(self):
        """Configure logging levels for third-party libraries."""
        noisy_loggers = {
            'httpx': logging.WARNING,
            'httpcore': logging.WARNING,
            'urllib3': logging.WARNING,
            'asyncio': logging.WARNING,
            'PIL': logging.WARNING,
            'matplotlib': logging.WARNING,
            'huggingface_hub': logging.WARNING,
            'diffusers': logging.WARNING,
            'transformers': logging.WARNING,
            'torch': logging.WARNING,
            'whisper': logging.WARNING,
            'aiosqlite': logging.WARNING,
            'aiogram': logging.INFO,
        }
        
        for logger_name, logger_level in noisy_loggers.items():
            logging.getLogger(logger_name).setLevel(logger_level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger with given name."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        return self.loggers[name]
    
    @contextmanager
    def request_context(self, request_id: str, **kwargs):
        """Context manager for request-scoped logging."""
        self.request_id_var.value = request_id
        self.request_id_var.context = kwargs
        
        # Log request start
        logger = self.get_logger(__name__)
        logger.info(f"Request started: {request_id}", extra={'request_start': True, **kwargs})
        
        try:
            yield
        finally:
            # Log request end
            logger.info(f"Request completed: {request_id}", extra={'request_end': True, **kwargs})
            
            # Clear context
            if hasattr(self.request_id_var, 'value'):
                delattr(self.request_id_var, 'value')
            if hasattr(self.request_id_var, 'context'):
                delattr(self.request_id_var, 'context')
    
    def log_metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, Any]] = None):
        """Log a metric value."""
        metric = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'name': name,
            'value': value,
            'tags': tags or {}
        }
        self.metrics_queue.put(metric)
    
    def _process_metrics(self):
        """Background processor for metrics."""
        logger = self.get_logger('metrics')
        batch = []
        
        while True:
            try:
                # Collect metrics in batches
                metric = self.metrics_queue.get(timeout=1)
                batch.append(metric)
                
                # Write batch if size threshold reached
                if len(batch) >= 10:
                    logger.log(LogLevel.METRIC.value, json.dumps(batch))
                    batch.clear()
                    
            except Empty:
                # Write remaining metrics
                if batch:
                    logger.log(LogLevel.METRIC.value, json.dumps(batch))
                    batch.clear()
            except Exception as e:
                print(f"Error processing metrics: {e}", file=sys.stderr)


# Global manager instance
_manager = LoggerManager()


def setup_logging(**kwargs) -> logging.Logger:
    """Setup logging system."""
    return _manager.setup_logging(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return _manager.get_logger(name)


@contextmanager
def log_context(**kwargs):
    """Context manager for adding context to all logs in scope."""
    # Get current context
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **factory_kwargs):
        record = old_factory(*args, **factory_kwargs)
        # Add context to record
        for key, value in kwargs.items():
            setattr(record, key, value)
        # Add request ID if available
        if hasattr(_manager.request_id_var, 'value'):
            record.request_id = _manager.request_id_var.value
        if hasattr(_manager.request_id_var, 'context'):
            for key, value in _manager.request_id_var.context.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    try:
        yield
    finally:
        logging.setLogRecordFactory(old_factory)


def log_execution_time(func: Optional[Callable] = None, *, 
                      level: int = logging.INFO,
                      message: Optional[str] = None):
    """Decorator to log function execution time."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger(f.__module__)
            
            try:
                result = await f(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                log_message = message or f"Function {f.__name__} executed"
                logger.log(level, f"{log_message} in {duration:.2f}ms")
                
                # Log metric
                _manager.log_metric(
                    f"function_duration_ms",
                    duration,
                    {'function': f.__name__, 'module': f.__module__}
                )
                
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"Function {f.__name__} failed after {duration:.2f}ms: {e}",
                    exc_info=True
                )
                raise
        
        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger(f.__module__)
            
            try:
                result = f(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                log_message = message or f"Function {f.__name__} executed"
                logger.log(level, f"{log_message} in {duration:.2f}ms")
                
                # Log metric
                _manager.log_metric(
                    f"function_duration_ms",
                    duration,
                    {'function': f.__name__, 'module': f.__module__}
                )
                
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"Function {f.__name__} failed after {duration:.2f}ms: {e}",
                    exc_info=True
                )
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper
    
    # Support both @decorator and @decorator() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def log_error(error_type: str = "unknown"):
    """Decorator to log errors with context."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(func.__module__)
                logger.error(
                    f"{error_type} error in {func.__name__}: {e}",
                    exc_info=True,
                    extra={
                        'error_type': error_type,
                        'function': func.__name__,
                        'module': func.__module__
                    }
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(func.__module__)
                logger.error(
                    f"{error_type} error in {func.__name__}: {e}",
                    exc_info=True,
                    extra={
                        'error_type': error_type,
                        'function': func.__name__,
                        'module': func.__module__
                    }
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Custom log methods
class EnhancedLogger(logging.Logger):
    """Logger with additional convenience methods."""
    
    def trace(self, message: str, *args, **kwargs):
        """Log at TRACE level."""
        self.log(LogLevel.TRACE.value, message, *args, **kwargs)
    
    def metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, Any]] = None):
        """Log a metric."""
        _manager.log_metric(name, value, tags)
    
    def struct(self, message: str, data: Dict[str, Any], *args, **kwargs):
        """Log with structured data."""
        kwargs['extra'] = {**(kwargs.get('extra', {})), **data}
        self.info(message, *args, **kwargs)


# Set custom logger class
logging.setLoggerClass(EnhancedLogger)


# Convenience exports
__all__ = [
    'setup_logging',
    'get_logger',
    'log_context',
    'log_execution_time',
    'log_error',
    'LogLevel',
    'EnhancedLogger'
]