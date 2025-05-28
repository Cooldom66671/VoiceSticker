import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Настройки Telegram Бота ---
BOT_TOKEN = os.getenv("BOT_TOKEN")

# --- Настройки хранения файлов ---
# Папка для временного хранения загруженных аудио и сгенерированных стикеров
STORAGE_DIR = "storage"
DATABASE_PATH = "stickers.db" # Путь к файлу базы данных SQLite