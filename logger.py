import logging
import os

# Создаем папку для логов, если ее нет
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Имя файла лога
LOG_FILE = os.path.join(LOG_DIR, "bot.log")

def setup_logging():
    """Настраивает логирование для приложения."""
    logging.basicConfig(
        level=logging.INFO, # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'), # Запись логов в файл
            logging.StreamHandler() # Вывод логов в консоль
        ]
    )
    # Уменьшаем шум от некоторых библиотек
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    logging.getLogger('diffusers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING) # Pillow тоже может быть шумным