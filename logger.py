"""
Модуль настройки логирования для Telegram-бота "Голостикеры".
Обеспечивает централизованное логирование с ротацией файлов и фильтрацией.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Импортируем настройки из config
try:
    from config import (
        LOGS_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT,
        LOG_FILE_MAX_BYTES, LOG_FILE_BACKUP_COUNT
    )
except ImportError:
    # Дефолтные значения, если config.py еще не готов
    LOGS_DIR = Path("logs")
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    LOG_FILE_BACKUP_COUNT = 5

# Создаем папку для логов, если ее нет
LOGS_DIR = Path(LOGS_DIR)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Имя файла лога с датой
LOG_FILE = LOGS_DIR / "bot.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли."""
    
    # ANSI escape коды для цветов
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись лога с цветом."""
        # Сохраняем оригинальный levelname
        levelname = record.levelname
        
        # Добавляем цвет к levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Форматируем сообщение
        formatted = super().format(record)
        
        # Восстанавливаем оригинальный levelname
        record.levelname = levelname
        
        return formatted


class ContextFilter(logging.Filter):
    """Фильтр для добавления контекстной информации в логи."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Добавляет дополнительную информацию в запись лога."""
        # Добавляем имя функции, если доступно
        if hasattr(record, 'funcName') and record.funcName != '<module>':
            record.funcName = f"[{record.funcName}]"
        else:
            record.funcName = ""
        
        # Добавляем ID пользователя, если есть в контексте
        if not hasattr(record, 'user_id'):
            record.user_id = ""
        
        return True


def setup_logging(
    log_level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    colored_output: bool = True
) -> logging.Logger:
    """
    Настраивает логирование для приложения.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Записывать ли логи в файл
        log_to_console: Выводить ли логи в консоль
        colored_output: Использовать ли цветной вывод в консоли
        
    Returns:
        Настроенный корневой логгер
    """
    # Определяем уровень логирования
    level = getattr(logging, log_level or LOG_LEVEL.upper())
    
    # Получаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Очищаем существующие обработчики
    root_logger.handlers.clear()
    
    # Создаем форматтер
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Добавляем контекстный фильтр
    context_filter = ContextFilter()
    
    # Настраиваем обработчик для файла
    if log_to_file:
        # Основной лог с ротацией
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)
        
        # Отдельный файл для ошибок
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)
    
    # Настраиваем обработчик для консоли
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Используем цветной форматтер для консоли
        if colored_output and sys.stdout.isatty():
            colored_formatter = ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(formatter)
        
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # Настраиваем уровни для шумных библиотек
    noisy_loggers = {
        'httpx': logging.WARNING,
        'httpcore': logging.WARNING,
        'huggingface_hub': logging.WARNING,
        'diffusers': logging.WARNING,
        'transformers': logging.WARNING,
        'asyncio': logging.WARNING,
        'PIL': logging.WARNING,
        'urllib3': logging.WARNING,
        'filelock': logging.WARNING,
        'torch': logging.WARNING,
        'whisper': logging.WARNING,
        'aiosqlite': logging.WARNING,
    }
    
    for logger_name, logger_level in noisy_loggers.items():
        logging.getLogger(logger_name).setLevel(logger_level)
    
    # Логируем информацию о запуске
    root_logger.info("=" * 60)
    root_logger.info(f"Запуск системы логирования - {datetime.now()}")
    root_logger.info(f"Уровень логирования: {log_level or LOG_LEVEL}")
    root_logger.info(f"Логи сохраняются в: {LOG_FILE}")
    root_logger.info("=" * 60)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Получает логгер с указанным именем.
    
    Args:
        name: Имя логгера (обычно __name__ модуля)
        
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger):
    """
    Декоратор для логирования вызовов функций.
    
    Args:
        logger: Логгер для записи
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Вызов функции {func.__name__} с args={args}, kwargs={kwargs}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Функция {func.__name__} завершена успешно")
                return result
            except Exception as e:
                logger.error(f"Ошибка в функции {func.__name__}: {e}", exc_info=True)
                raise
        
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Вызов функции {func.__name__} с args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Функция {func.__name__} завершена успешно")
                return result
            except Exception as e:
                logger.error(f"Ошибка в функции {func.__name__}: {e}", exc_info=True)
                raise
        
        # Возвращаем правильный wrapper в зависимости от типа функции
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class LogContext:
    """Контекстный менеджер для добавления информации в логи."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self._old_factory = None
    
    def __enter__(self):
        """Входим в контекст."""
        self._old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выходим из контекста."""
        logging.setLogRecordFactory(self._old_factory)


# Создаем дефолтный логгер при импорте модуля
if __name__ != "__main__":
    setup_logging()