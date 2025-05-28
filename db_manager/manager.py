import aiosqlite
import logging
import os # Для os.path.basename

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None # Соединение с базой данных

    async def _get_connection(self):
        """Получает или создает соединение с базой данных."""
        if self.conn is None:
            self.conn = await aiosqlite.connect(self.db_path)
            logger.info(f"Соединение с базой данных '{self.db_path}' установлено.")
        return self.conn

    async def create_table(self):
        """Создает таблицу стикеров, если она не существует."""
        conn = await self._get_connection()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stickers (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    prompt TEXT,
                    file_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.commit()
            logger.info("Таблица 'stickers' проверена/создана.")
        except Exception as e:
            logger.error(f"Ошибка при создании таблицы: {e}")

    async def add_sticker(self, user_id: int, prompt: str, file_path: str):
        """Добавляет информацию о сгенерированном стикере в базу данных."""
        conn = await self._get_connection()
        # Используем уникальный идентификатор файла как ID стикера
        sticker_id = os.path.basename(file_path).split('.')[0]
        try:
            await conn.execute(
                "INSERT INTO stickers (id, user_id, prompt, file_path) VALUES (?, ?, ?, ?)",
                (sticker_id, user_id, prompt, file_path)
            )
            await conn.commit()
            logger.info(f"Стикер '{sticker_id}' для пользователя {user_id} добавлен в БД.")
        except Exception as e:
            logger.error(f"Ошибка при добавлении стикера в БД: {e}")

    async def get_user_stickers(self, user_id: int):
        """Получает все стикеры, сгенерированные пользователем."""
        conn = await self._get_connection()
        try:
            cursor = await conn.execute("SELECT id, prompt, file_path, timestamp FROM stickers WHERE user_id = ?", (user_id,))
            stickers = await cursor.fetchall()
            logger.info(f"Получено {len(stickers)} стикеров для пользователя {user_id}.")
            return stickers
        except Exception as e:
            logger.error(f"Ошибка при получении стикеров пользователя {user_id}: {e}")
            return []

    async def close_connection(self):
        """Закрывает соединение с базой данных."""
        if self.conn:
            await self.conn.close()
            self.conn = None
            logger.info("Соединение с базой данных закрыто.")