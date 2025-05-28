"""
Конфигурация для Telegram-бота "Голостикеры".
Централизованное управление настройками приложения.
"""
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Базовые пути ---
# Корневая директория проекта
BASE_DIR = Path(__file__).resolve().parent
# Директория для хранения данных
DATA_DIR = BASE_DIR / "data"

# --- Настройки Telegram Бота ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    print("ОШИБКА: BOT_TOKEN не найден в переменных окружения!")
    print("Создайте файл .env и добавьте туда BOT_TOKEN=your_token_here")
    sys.exit(1)

# --- Настройки хранения файлов ---
# Папка для временного хранения загруженных аудио и сгенерированных стикеров
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
if not STORAGE_DIR.is_absolute():
    STORAGE_DIR = BASE_DIR / STORAGE_DIR

# Папка для логов
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
if not LOGS_DIR.is_absolute():
    LOGS_DIR = BASE_DIR / LOGS_DIR

# Путь к файлу базы данных SQLite
DB_PATH = Path(os.getenv("DB_PATH", "stickers.db"))
if not DB_PATH.is_absolute():
    DB_PATH = BASE_DIR / DB_PATH

# --- Настройки моделей ---
# Whisper
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu, cuda, mps

# Stable Diffusion
STABLE_DIFFUSION_MODEL = os.getenv("STABLE_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5")
SD_DEVICE = os.getenv("SD_DEVICE", "mps")  # cpu, cuda, mps
SD_DTYPE = os.getenv("SD_DTYPE", "float16")  # float16, float32

# --- Настройки генерации ---
# Параметры для Stable Diffusion
SD_NUM_INFERENCE_STEPS = int(os.getenv("SD_NUM_INFERENCE_STEPS", "30"))
SD_GUIDANCE_SCALE = float(os.getenv("SD_GUIDANCE_SCALE", "7.5"))
SD_HEIGHT = int(os.getenv("SD_HEIGHT", "512"))
SD_WIDTH = int(os.getenv("SD_WIDTH", "512"))

# --- Настройки стикеров ---
STICKER_MAX_SIZE = 512  # Максимальный размер стикера в пикселях
STICKER_FORMAT = "PNG"  # Формат файла стикера
STICKER_QUALITY = 95  # Качество сжатия (для JPEG)

# --- Настройки безопасности ---
MAX_AUDIO_SIZE_MB = 20  # Максимальный размер аудио файла в МБ
MAX_AUDIO_DURATION_SEC = 60  # Максимальная длительность аудио в секундах
ALLOWED_AUDIO_FORMATS = {".ogg", ".oga", ".mp3", ".wav", ".m4a"}

# --- Настройки производительности ---
# Таймауты в секундах
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "30"))
SD_TIMEOUT = int(os.getenv("SD_TIMEOUT", "60"))
DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", "5"))

# --- Настройки логирования ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5

# --- Настройки для пользователя ---
# Тексты сообщений
MESSAGES = {
    "start": (
        "👋 Привет! Я бот *Голостикеры*!\n\n"
        "🎤 Отправь мне голосовое сообщение, и я создам уникальный стикер на основе твоих слов.\n"
        "✍️ Или просто напиши текст, и я превращу его в стикер!\n\n"
        "Используй /help для подробной информации."
    ),
    "help": (
        "🤖 *Как использовать бота:*\n\n"
        "1️⃣ *Голосовое сообщение:*\n"
        "   • Запиши голосовое сообщение\n"
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
    "transcription_result": "📝 Распознанный текст: *{}*\n\n🎨 Генерирую стикер...",
    "processing_text": "🎨 Генерирую стикер по вашему описанию...",
    "sticker_ready": "✅ Стикер готов! Вы можете добавить его в свою коллекцию.",
    "error_transcription": "❌ Не удалось распознать голосовое сообщение. Попробуйте еще раз.",
    "error_generation": "❌ Не удалось создать стикер. Попробуйте другое описание.",
    "error_processing": "❌ Произошла ошибка при обработке. Попробуйте позже.",
    "error_file_too_large": "❌ Файл слишком большой. Максимальный размер: {} МБ",
    "error_invalid_format": "❌ Неподдерживаемый формат файла.",
}

# --- Вспомогательные функции ---
def get_device_config():
    """Возвращает оптимальную конфигурацию устройств для текущей системы."""
    import torch
    
    # Проверяем доступность устройств
    if torch.cuda.is_available() and SD_DEVICE == "cuda":
        return {
            "whisper_device": "cuda",
            "sd_device": "cuda",
            "sd_dtype": torch.float16 if SD_DTYPE == "float16" else torch.float32
        }
    elif torch.backends.mps.is_available() and SD_DEVICE == "mps":
        return {
            "whisper_device": "cpu",  # Whisper лучше работает на CPU для Apple Silicon
            "sd_device": "mps",
            "sd_dtype": torch.float16 if SD_DTYPE == "float16" else torch.float32
        }
    else:
        return {
            "whisper_device": "cpu",
            "sd_device": "cpu",
            "sd_dtype": torch.float32  # float16 не поддерживается на CPU
        }

# Создаем необходимые директории при импорте
for directory in [STORAGE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)