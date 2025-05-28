import os
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties # !!! Новый импорт для aiogram 3.7+ !!!

# Импортируем наши модули
from config import BOT_TOKEN, STORAGE_DIR, DATABASE_PATH
from logger import setup_logging
from db_manager.manager import DatabaseManager
from bot_handlers import register_handlers # Мы импортируем функцию, которую скоро напишем

# Для загрузки моделей Whisper и Stable Diffusion при старте
from stt_service.service import load_whisper_model
from image_generation_service.service import load_stable_diffusion_pipeline

# Настраиваем логирование
setup_logging()
import logging
logger = logging.getLogger(__name__)

async def main():
    """Основная функция для запуска бота."""
    # 1. Проверяем наличие токена бота
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не найден. Убедитесь, что он указан в файле .env")
        return

    # 2. Создаем папку для хранения файлов, если она не существует
    os.makedirs(STORAGE_DIR, exist_ok=True)
    logger.info(f"Папка для хранения файлов '{STORAGE_DIR}' проверена/создана.")

    # 3. Инициализируем базу данных
    db_manager = DatabaseManager(DATABASE_PATH)
    await db_manager.create_table()
    logger.info(f"База данных '{DATABASE_PATH}' инициализирована.")

    # 4. Инициализируем объекты Bot и Dispatcher
    # Исправлено согласно требованиям aiogram 3.7+
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    dp = Dispatcher()

    # Передаем объекты (logger, db_manager) в контекст aiogram,
    # чтобы они были доступны в обработчиках
    dp["logger"] = logger
    dp["db_manager"] = db_manager
    dp["storage_dir"] = STORAGE_DIR
    dp["bot_instance"] = bot # Передаем сам объект бота

    # 5. Загружаем модели Whisper и Stable Diffusion заранее
    # Это может занять некоторое время при первом запуске
    logger.info("Начинаю предварительную загрузку моделей...")
    await load_whisper_model()
    await load_stable_diffusion_pipeline()
    logger.info("Все модели успешно загружены.")

    # 6. Регистрируем обработчики сообщений
    register_handlers(dp)
    logger.info("Обработчики сообщений зарегистрированы.")

    # 7. Запускаем бота
    try:
        logger.info("Бот запускается...")
        # Skip_updates=True для пропуска старых обновлений при запуске
        await dp.start_polling(bot, skip_updates=True) 
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
    finally:
        # Закрываем соединение с базой данных при завершении работы бота
        await db_manager.close_connection()
        logger.info("Бот остановлен. Соединение с базой данных закрыто.")


if __name__ == "__main__":
    # Запускаем асинхронную функцию main
    asyncio.run(main())