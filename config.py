"""
Configuration module for Telegram bot "Golostickery".
Provides centralized configuration management with validation and type safety.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeviceType(Enum):
    """Supported device types for ML models."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class ModelSize(Enum):
    """Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class PathConfig:
    """File system paths configuration."""
    base_dir: Path
    storage_dir: Path
    logs_dir: Path
    cache_dir: Path
    models_dir: Path
    db_path: Path
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr_name in ['storage_dir', 'logs_dir', 'cache_dir', 'models_dir']:
            dir_path = getattr(self, attr_name)
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str
    webhook_url: Optional[str] = None
    webhook_port: int = 8443
    use_webhook: bool = False
    allowed_updates: List[str] = field(default_factory=lambda: ["message", "callback_query"])
    drop_pending_updates: bool = True
    
    def __post_init__(self):
        """Validate Telegram configuration."""
        if not self.bot_token:
            raise ValueError("BOT_TOKEN is required")
        
        if not self.bot_token.count(':') == 1:
            raise ValueError("Invalid bot token format")
        
        bot_id, bot_hash = self.bot_token.split(':')
        if not bot_id.isdigit() or len(bot_hash) != 35:
            raise ValueError("Invalid bot token structure")


@dataclass
class WhisperConfig:
    """Whisper model configuration."""
    model_size: ModelSize
    device: DeviceType
    language: Optional[str] = None
    initial_prompt: Optional[str] = None
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    
    def get_model_name(self) -> str:
        """Get the full model name."""
        return self.model_size.value


@dataclass
class StableDiffusionConfig:
    """Stable Diffusion configuration."""
    model_id: str
    device: DeviceType
    dtype: torch.dtype
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    use_xformers: bool = False
    scheduler: str = "DPMSolverMultistepScheduler"
    
    def __post_init__(self):
        """Validate SD configuration."""
        if self.num_inference_steps < 1 or self.num_inference_steps > 100:
            raise ValueError("num_inference_steps must be between 1 and 100")
        
        if self.guidance_scale < 1.0 or self.guidance_scale > 20.0:
            raise ValueError("guidance_scale must be between 1.0 and 20.0")
        
        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError("height and width must be divisible by 8")


@dataclass
class StickerConfig:
    """Sticker processing configuration."""
    max_size: int = 512
    format: str = "PNG"
    quality: int = 95
    max_file_size_kb: int = 350
    background_removal_threshold: int = 240
    shadow_offset: tuple[int, int] = (5, 5)
    shadow_blur_radius: int = 5
    shadow_opacity: int = 128
    outline_size: int = 3
    outline_color: tuple[int, int, int, int] = (255, 255, 255, 255)


@dataclass
class SecurityConfig:
    """Security and limits configuration."""
    max_audio_size_mb: float = 20.0
    max_audio_duration_sec: int = 60
    allowed_audio_formats: set[str] = field(
        default_factory=lambda: {".ogg", ".oga", ".mp3", ".wav", ".m4a"}
    )
    rate_limit_messages_per_minute: int = 20
    rate_limit_messages_per_hour: int = 100
    max_concurrent_generations: int = 2  # ИСПРАВЛЕНО: перенесено сюда
    allowed_user_ids: Optional[List[int]] = None
    blocked_user_ids: List[int] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Performance and timeout configuration."""
    whisper_timeout_sec: int = 30
    sd_timeout_sec: int = 60
    db_timeout_sec: int = 5
    cleanup_interval_hours: int = 24
    cleanup_keep_days: int = 7
    cache_ttl_hours: int = 24
    max_workers: int = 4
    use_memory_efficient_attention: bool = True
    offload_to_cpu: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    file_backup_count: int = 5
    log_sql_queries: bool = False
    log_performance_metrics: bool = True
    colored_output: bool = True


@dataclass
class MessageTemplates:
    """Bot message templates with multi-language support."""
    language: str = "ru"
    templates: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Load message templates."""
        self.templates = {
            "ru": {
                "start": (
                    "👋 Привет, {name}! Я бот *Голостикеры*!\n\n"
                    "🎤 Отправь мне голосовое сообщение, и я создам уникальный стикер на основе твоих слов.\n"
                    "✍️ Или просто напиши текст, и я превращу его в стикер!\n\n"
                    "Используй /help для подробной информации."
                ),
                "help": (
                    "🤖 *Как использовать бота:*\n\n"
                    "1️⃣ *Голосовое сообщение:*\n"
                    "   • Запиши голосовое сообщение с описанием\n"
                    "   • Я распознаю твою речь\n"
                    "   • Создам стикер по твоим словам\n\n"
                    "2️⃣ *Текстовое сообщение:*\n"
                    "   • Напиши описание стикера\n"
                    "   • Я сгенерирую изображение\n\n"
                    "📝 *Примеры запросов:*\n"
                    "   • «Милый котик с радугой»\n"
                    "   • «Космонавт на Луне»\n"
                    "   • «Пицца с улыбкой»\n\n"
                    "⚡ *Советы:*\n"
                    "   • Будь конкретным в описании\n"
                    "   • Используй прилагательные\n"
                    "   • Описывай эмоции и действия\n\n"
                    "🎨 Генерация занимает 10-30 секунд"
                ),
                "processing_voice": "🎤 Распознаю голосовое сообщение...",
                "transcription_result": "📝 Распознанный текст: *{}*",
                "processing_text": "🎨 Генерирую стикер по вашему описанию...",
                "sticker_ready": "✅ Стикер готов!",
                "error_transcription": "❌ Не удалось распознать голосовое сообщение. Попробуйте еще раз.",
                "error_generation": "❌ Не удалось создать стикер. Попробуйте другое описание.",
                "error_processing": "❌ Произошла ошибка при обработке. Попробуйте позже.",
                "error_file_too_large": "❌ Файл слишком большой. Максимальный размер: {} МБ",
                "error_invalid_format": "❌ Неподдерживаемый формат файла.",
                "error_rate_limit": "⏳ Слишком много запросов. Подождите немного.",
                "error_maintenance": "🔧 Бот на техническом обслуживании. Попробуйте позже.",
                "stats_header": "📊 *Ваша статистика:*\n\n",
                "style_selection": "🎨 Выберите стиль для вашего стикера:",
                "background_selection": "🖼 Выберите фон для стикера:",
                "generation_progress": "Генерация: {}%",
                "queue_position": "🔄 Ваша позиция в очереди: {}",
            },
            "en": {
                "start": (
                    "👋 Hello, {name}! I'm the *VoiceStickers* bot!\n\n"
                    "🎤 Send me a voice message, and I'll create a unique sticker based on your words.\n"
                    "✍️ Or just type text, and I'll turn it into a sticker!\n\n"
                    "Use /help for more information."
                ),
                # ... other English translations
            }
        }
    
    def get(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Get message template with formatting."""
        lang = language or self.language
        template = self.templates.get(lang, {}).get(key, "")
        
        if not template and lang != "en":
            # Fallback to English
            template = self.templates.get("en", {}).get(key, "")
        
        if not template:
            return f"[Missing template: {key}]"
        
        return template.format(**kwargs) if kwargs else template


class Config:
    """Main configuration class."""
    
    def __init__(self):
        """Initialize configuration from environment and defaults."""
        # Determine environment
        self.environment = Environment(os.getenv("ENVIRONMENT", "production"))
        
        # Initialize paths
        base_dir = Path(__file__).resolve().parent
        self.paths = PathConfig(
            base_dir=base_dir,
            storage_dir=self._get_path("STORAGE_DIR", base_dir / "storage"),
            logs_dir=self._get_path("LOGS_DIR", base_dir / "logs"),
            cache_dir=self._get_path("CACHE_DIR", base_dir / ".cache"),
            models_dir=self._get_path("MODELS_DIR", base_dir / "models"),
            db_path=self._get_path("DB_PATH", base_dir / "stickers.db")
        )
        
        # Initialize Telegram config
        self.telegram = TelegramConfig(
            bot_token=self._get_required_env("BOT_TOKEN"),
            webhook_url=os.getenv("WEBHOOK_URL"),
            webhook_port=int(os.getenv("WEBHOOK_PORT", "8443")),
            use_webhook=os.getenv("USE_WEBHOOK", "false").lower() == "true"
        )
        
        # Detect device configuration
        device_config = self._detect_device_config()
        
        # Initialize Whisper config
        self.whisper = WhisperConfig(
            model_size=ModelSize(os.getenv("WHISPER_MODEL", "base")),
            device=device_config["whisper_device"],
            language=os.getenv("WHISPER_LANGUAGE"),
            temperature=float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
        )
        
        # Initialize Stable Diffusion config
        self.stable_diffusion = StableDiffusionConfig(
            model_id=os.getenv("STABLE_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5"),
            device=device_config["sd_device"],
            dtype=device_config["sd_dtype"],
            num_inference_steps=int(os.getenv("SD_NUM_INFERENCE_STEPS", "30")),
            guidance_scale=float(os.getenv("SD_GUIDANCE_SCALE", "7.5")),
            height=int(os.getenv("SD_HEIGHT", "512")),
            width=int(os.getenv("SD_WIDTH", "512")),
            use_xformers=device_config["use_xformers"]
        )
        
        # Initialize sticker config
        self.sticker = StickerConfig()
        
        # Initialize security config
        self.security = SecurityConfig(
            max_audio_size_mb=float(os.getenv("MAX_AUDIO_SIZE_MB", "20.0")),
            max_audio_duration_sec=int(os.getenv("MAX_AUDIO_DURATION_SEC", "60")),
            allowed_user_ids=self._parse_int_list(os.getenv("ALLOWED_USER_IDS")),
            blocked_user_ids=self._parse_int_list(os.getenv("BLOCKED_USER_IDS"))
        )
        
        # Initialize performance config
        self.performance = PerformanceConfig(
            whisper_timeout_sec=int(os.getenv("WHISPER_TIMEOUT", "30")),
            sd_timeout_sec=int(os.getenv("SD_TIMEOUT", "60")),
            db_timeout_sec=int(os.getenv("DB_TIMEOUT", "5")),
            max_workers=min(os.cpu_count() or 4, 8),
            use_memory_efficient_attention=device_config["use_memory_efficient_attention"],
            offload_to_cpu=device_config["offload_to_cpu"]
        )
        
        # Initialize logging config
        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_sql_queries=os.getenv("LOG_SQL_QUERIES", "false").lower() == "true",
            log_performance_metrics=os.getenv("LOG_PERFORMANCE_METRICS", "true").lower() == "true"
        )
        
        # Initialize message templates
        self.messages = MessageTemplates(
            language=os.getenv("BOT_LANGUAGE", "ru")
        )
        
        # Validate complete configuration
        self._validate_config()
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable."""
        value = os.getenv(key)
        if not value:
            print(f"ERROR: {key} not found in environment variables!")
            print(f"Please create .env file and add {key}=your_value")
            sys.exit(1)
        return value
    
    def _get_path(self, env_key: str, default: Path) -> Path:
        """Get path from environment or default."""
        path_str = os.getenv(env_key)
        if path_str:
            path = Path(path_str)
            return path if path.is_absolute() else Path(__file__).parent / path
        return default
    
    def _parse_int_list(self, value: Optional[str]) -> Optional[List[int]]:
        """Parse comma-separated list of integers."""
        if not value:
            return None
        try:
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            return None
    
    def _detect_device_config(self) -> Dict[str, Any]:
        """Detect optimal device configuration."""
        config = {
            "whisper_device": DeviceType.CPU,
            "sd_device": DeviceType.CPU,
            "sd_dtype": torch.float32,
            "use_xformers": False,
            "use_memory_efficient_attention": True,
            "offload_to_cpu": False
        }
        
        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_device = os.getenv("CUDA_DEVICE", "cuda")
            config["whisper_device"] = DeviceType.CUDA
            config["sd_device"] = DeviceType.CUDA
            config["sd_dtype"] = torch.float16
            
            # Check for xformers
            try:
                import xformers
                config["use_xformers"] = True
            except ImportError:
                pass
            
            # Check available memory
            if torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:  # Less than 6GB
                config["offload_to_cpu"] = True
        
        # Check MPS availability (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config["whisper_device"] = DeviceType.CPU  # Whisper works better on CPU for M1/M2
            config["sd_device"] = DeviceType.MPS
            config["sd_dtype"] = torch.float16
        
        # Override with environment variables if specified
        if os.getenv("WHISPER_DEVICE"):
            config["whisper_device"] = DeviceType(os.getenv("WHISPER_DEVICE"))
        
        if os.getenv("SD_DEVICE"):
            config["sd_device"] = DeviceType(os.getenv("SD_DEVICE"))
        
        if os.getenv("SD_DTYPE"):
            dtype_map = {"float16": torch.float16, "float32": torch.float32}
            config["sd_dtype"] = dtype_map.get(os.getenv("SD_DTYPE"), torch.float32)
        
        return config
    
    def _validate_config(self):
        """Validate complete configuration."""
        # Check FFmpeg availability
        import shutil
        if not shutil.which("ffmpeg"):
            print("WARNING: ffmpeg not found. Audio processing may not work correctly.")
            print("Install ffmpeg: https://ffmpeg.org/download.html")
        
        # Check model compatibility
        if self.stable_diffusion.device == DeviceType.CPU and self.stable_diffusion.dtype == torch.float16:
            print("WARNING: float16 is not supported on CPU. Switching to float32.")
            self.stable_diffusion.dtype = torch.float32
        
        # Validate paths
        if not os.access(self.paths.base_dir, os.W_OK):
            print(f"ERROR: No write access to base directory: {self.paths.base_dir}")
            sys.exit(1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "paths": {
                "base_dir": str(self.paths.base_dir),
                "storage_dir": str(self.paths.storage_dir),
                "logs_dir": str(self.paths.logs_dir),
                "cache_dir": str(self.paths.cache_dir),
                "models_dir": str(self.paths.models_dir),
                "db_path": str(self.paths.db_path)
            },
            "telegram": {
                "use_webhook": self.telegram.use_webhook,
                "webhook_port": self.telegram.webhook_port
            },
            "whisper": {
                "model": self.whisper.model_size.value,
                "device": self.whisper.device.value
            },
            "stable_diffusion": {
                "model": self.stable_diffusion.model_id,
                "device": self.stable_diffusion.device.value,
                "steps": self.stable_diffusion.num_inference_steps
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "timeouts": {
                    "whisper": self.performance.whisper_timeout_sec,
                    "sd": self.performance.sd_timeout_sec,
                    "db": self.performance.db_timeout_sec
                }
            }
        }
    
    def save_to_file(self, filepath: Path):
        """Save configuration to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'Config':
        """Load configuration from JSON file."""
        # This would require implementing a more complex loading mechanism
        # For now, we'll use environment-based config
        return cls()


# Create global configuration instance
config = Config()

# Export commonly used values for backward compatibility
BOT_TOKEN = config.telegram.bot_token
STORAGE_DIR = config.paths.storage_dir
LOGS_DIR = config.paths.logs_dir
DB_PATH = config.paths.db_path
WHISPER_MODEL = config.whisper.get_model_name()
WHISPER_DEVICE = config.whisper.device.value
STABLE_DIFFUSION_MODEL = config.stable_diffusion.model_id
SD_DEVICE = config.stable_diffusion.device.value
SD_DTYPE = config.stable_diffusion.dtype
SD_NUM_INFERENCE_STEPS = config.stable_diffusion.num_inference_steps
SD_GUIDANCE_SCALE = config.stable_diffusion.guidance_scale
SD_HEIGHT = config.stable_diffusion.height
SD_WIDTH = config.stable_diffusion.width
STICKER_MAX_SIZE = config.sticker.max_size
STICKER_FORMAT = config.sticker.format
STICKER_QUALITY = config.sticker.quality
MAX_AUDIO_SIZE_MB = config.security.max_audio_size_mb
MAX_AUDIO_DURATION_SEC = config.security.max_audio_duration_sec
ALLOWED_AUDIO_FORMATS = config.security.allowed_audio_formats
WHISPER_TIMEOUT = config.performance.whisper_timeout_sec
SD_TIMEOUT = config.performance.sd_timeout_sec
DB_TIMEOUT = config.performance.db_timeout_sec
LOG_LEVEL = config.logging.level
LOG_FORMAT = config.logging.format
LOG_DATE_FORMAT = config.logging.date_format
LOG_FILE_MAX_BYTES = config.logging.file_max_bytes
LOG_FILE_BACKUP_COUNT = config.logging.file_backup_count
MESSAGES = {k: v for k, v in config.messages.templates.get(config.messages.language, {}).items()}

# Helper function for backward compatibility
def get_device_config() -> Dict[str, Any]:
    """Get device configuration for ML models."""
    return {
        "whisper_device": config.whisper.device.value,
        "sd_device": config.stable_diffusion.device.value,
        "sd_dtype": config.stable_diffusion.dtype
    }