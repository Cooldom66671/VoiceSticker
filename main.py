"""
Главный модуль Telegram-бота "Голостикеры".
Точка входа в приложение, инициализация всех компонентов.
"""
import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

# Импортируем наши модули
from config import (
    BOT_TOKEN, STORAGE_DIR, DB_PATH, LOGS_DIR,
    WHISPER_TIMEOUT, SD_TIMEOUT, get_device_config
)
from logger import setup_logging, get_logger
from db_manager.manager import DatabaseManager
from bot_handlers import register_handlers

# Для предварительной загрузки моделей
from stt_service.service import load_whisper_model
from image_generation_service.service import load_stable_diffusion_pipeline

# Настраиваем логирование
setup_logging()
logger = get_logger(__name__)

# Глобальная переменная для graceful shutdown
should_exit = False


class BotApplication:
    """Класс для управления жизненным циклом бота."""
    
    def __init__(self):
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self.db_manager: Optional[DatabaseManager] = None
        self._running = False
    
    async def setup(self):
        """Инициализация всех компонентов бота."""
        logger.info("=" * 60)
        logger.info("Запуск Telegram-бота 'Голостикеры'")
        logger.info("=" * 60)
        
        # 1. Проверяем конфигурацию
        self._check_configuration()
        
        # 2. Создаем необходимые директории
        self._create_directories()
        
        # 3. Инициализируем базу данных
        await self._init_database()
        
        # 4. Инициализируем бота и диспетчер
        self._init_bot()
        
        # 5. Загружаем ML модели
        await self._load_models()
        
        # 6. Регистрируем обработчики
        self._register_handlers()
        
        logger.info("Инициализация завершена успешно")
    
    def _check_configuration(self):
        """Проверяет корректность конфигурации."""
        logger.info("Проверка конфигурации...")
        
        # BOT_TOKEN уже проверяется в config.py
        # Дополнительные проверки при необходимости
        
        # Проверяем доступность ffmpeg для работы с аудио
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise Exception("ffmpeg не работает корректно")
            logger.info("✓ ffmpeg найден и работает")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(
                "⚠ ffmpeg не найден или не работает. "
                "Установите ffmpeg для корректной работы с аудио: "
                "https://ffmpeg.org/download.html"
            )
        
        # Проверяем конфигурацию устройств
        device_config = get_device_config()
        logger.info(f"✓ Конфигурация устройств: {device_config}")
    
    def _create_directories(self):
        """Создает необходимые директории."""
        logger.info("Создание директорий...")
        
        directories = {
            "storage": STORAGE_DIR,
            "logs": LOGS_DIR,
            "models_cache": Path(".cache/models"),
        }
        
        for name, path in directories.items():
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Директория '{name}': {path}")
    
    async def _init_database(self):
        """Инициализирует базу данных."""
        logger.info("Инициализация базы данных...")
        
        self.db_manager = DatabaseManager(str(DB_PATH))
        await self.db_manager.create_table()
        
        # Проверяем работу БД
        stats = await self.db_manager.get_stats()
        logger.info(f"✓ База данных инициализирована. Записей в БД: {stats.get('total_stickers', 0)}")
    
    def _init_bot(self):
        """Инициализирует бота и диспетчер."""
        logger.info("Инициализация бота...")
        
        # Создаем бота с настройками по умолчанию
        self.bot = Bot(
            token=BOT_TOKEN, 
            default=DefaultBotProperties(
                parse_mode=ParseMode.MARKDOWN,
                link_preview_is_disabled=True
            )
        )
        
        # Создаем диспетчер с хранилищем состояний
        self.dp = Dispatcher(storage=MemoryStorage())
        
        # Передаем зависимости в контекст диспетчера
        self.dp["logger"] = logger
        self.dp["db_manager"] = self.db_manager
        self.dp["storage_dir"] = str(STORAGE_DIR)
        self.dp["bot_instance"] = self.bot
        
        logger.info("✓ Бот и диспетчер инициализированы")
    
    async def _load_models(self):
        """Загружает ML модели."""
        logger.info("Загрузка ML моделей (это может занять несколько минут при первом запуске)...")
        
        try:
            # Загружаем Whisper с таймаутом
            logger.info("• Загрузка модели Whisper...")
            whisper_task = asyncio.create_task(load_whisper_model())
            await asyncio.wait_for(whisper_task, timeout=WHISPER_TIMEOUT)
            logger.info("✓ Модель Whisper загружена")
            
            # Загружаем Stable Diffusion с таймаутом
            logger.info("• Загрузка модели Stable Diffusion...")
            sd_task = asyncio.create_task(load_stable_diffusion_pipeline())
            await asyncio.wait_for(sd_task, timeout=SD_TIMEOUT)
            logger.info("✓ Модель Stable Diffusion загружена")
            
        except asyncio.TimeoutError:
            logger.error("Превышено время ожидания загрузки моделей")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке моделей: {e}")
            raise
    
    def _register_handlers(self):
        """Регистрирует обработчики сообщений."""
        logger.info("Регистрация обработчиков...")
        
        register_handlers(self.dp)
        
        # Регистрируем обработчики для graceful shutdown
        self.dp.startup.register(self._on_startup)
        self.dp.shutdown.register(self._on_shutdown)
        
        logger.info("✓ Обработчики зарегистрированы")
    
    async def _on_startup(self):
        """Обработчик события запуска бота."""
        self._running = True
        logger.info("🚀 Бот успешно запущен и готов к работе!")
        
        # Отправляем уведомление администратору (если настроено)
        # await self._notify_admin("Бот запущен")
    
    async def _on_shutdown(self):
        """Обработчик события остановки бота."""
        self._running = False
        logger.info("Начинаю процедуру остановки бота...")
        
        # Закрываем соединения
        if self.db_manager:
            await self.db_manager.close_connection()
            logger.info("✓ Соединение с БД закрыто")
        
        # Закрываем сессию бота
        if self.bot:
            await self.bot.session.close()
            logger.info("✓ Сессия бота закрыта")
        
        logger.info("👋 Бот остановлен")
    
    async def run(self):
        """Запускает бота."""
        try:
            await self.setup()
            
            logger.info("Запуск polling...")
            await self.dp.start_polling(
                self.bot, 
                skip_updates=True,
                allowed_updates=["message", "callback_query"]
            )
            
        except KeyboardInterrupt:
            logger.info("Получен сигнал прерывания (Ctrl+C)")
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}", exc_info=True)
            raise
        finally:
            # Вызываем shutdown вручную, если он не был вызван
            if self._running:
                await self._on_shutdown()


def setup_signal_handlers(app: BotApplication):
    """Настраивает обработчики системных сигналов."""
    
    def signal_handler(sig, frame):
        global should_exit
        should_exit = True
        logger.info(f"Получен сигнал {sig}. Завершаю работу...")
        # Создаем задачу для корректной остановки
        asyncio.create_task(app._on_shutdown())
        sys.exit(0)
    
    # Регистрируем обработчики для graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Для Windows
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal_handler)


async def main():
    """Главная функция приложения."""
    app = BotApplication()
    setup_signal_handlers(app)
    
    try:
        await app.run()
    except Exception as e:
        logger.critical(f"Фатальная ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Настраиваем event loop для лучшей производительности
    if sys.platform == "win32":
        # Для Windows используем ProactorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Запускаем приложение
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Программа прервана пользователем")
    except Exception as e:
        logger.critical(f"Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)