"""
Модуль для работы с базой данных SQLite.
Управляет хранением метаданных о сгенерированных стикерах.
"""
import aiosqlite
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
import uuid

from logger import get_logger
from config import DB_TIMEOUT

logger = get_logger(__name__)


class DatabaseError(Exception):
    """Базовое исключение для ошибок базы данных."""
    pass


class DatabaseManager:
    """Менеджер для работы с базой данных стикеров."""
    
    def __init__(self, db_path: str):
        """
        Инициализирует менеджер базы данных.
        
        Args:
            db_path: Путь к файлу базы данных SQLite
        """
        self.db_path = Path(db_path)
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()  # Для потокобезопасности
        
        # Убеждаемся, что директория для БД существует
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @asynccontextmanager
    async def _get_connection(self):
        """
        Контекстный менеджер для получения соединения с БД.
        Автоматически управляет блокировками и обработкой ошибок.
        """
        async with self._lock:
            if self._connection is None:
                try:
                    self._connection = await aiosqlite.connect(
                        self.db_path,
                        timeout=DB_TIMEOUT
                    )
                    # Включаем foreign keys и оптимизации
                    await self._connection.execute("PRAGMA foreign_keys = ON")
                    await self._connection.execute("PRAGMA journal_mode = WAL")
                    await self._connection.execute("PRAGMA synchronous = NORMAL")
                    
                    logger.info(f"Соединение с БД установлено: {self.db_path}")
                except Exception as e:
                    logger.error(f"Ошибка подключения к БД: {e}")
                    raise DatabaseError(f"Не удалось подключиться к БД: {e}")
            
            try:
                yield self._connection
            except aiosqlite.Error as e:
                logger.error(f"Ошибка при работе с БД: {e}")
                raise DatabaseError(f"Ошибка базы данных: {e}")
    
    async def create_table(self):
        """Создает таблицы в базе данных."""
        async with self._get_connection() as conn:
            try:
                # Основная таблица стикеров
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS stickers (
                        id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        username TEXT,
                        prompt TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER,
                        generation_time REAL,
                        model_version TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        CHECK (user_id > 0),
                        CHECK (length(prompt) > 0),
                        CHECK (file_size >= 0)
                    )
                """)
                
                # Индексы для быстрого поиска
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_id 
                    ON stickers(user_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON stickers(created_at DESC)
                """)
                
                # Таблица статистики пользователей
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_stats (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT,
                        total_stickers INTEGER DEFAULT 0,
                        total_voice_requests INTEGER DEFAULT 0,
                        total_text_requests INTEGER DEFAULT 0,
                        first_use TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_use TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        CHECK (total_stickers >= 0),
                        CHECK (total_voice_requests >= 0),
                        CHECK (total_text_requests >= 0)
                    )
                """)
                
                # Таблица для хранения ошибок (для анализа)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS error_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        error_type TEXT NOT NULL,
                        error_message TEXT,
                        prompt TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await conn.commit()
                logger.info("Таблицы БД успешно созданы/проверены")
                
            except Exception as e:
                await conn.rollback()
                logger.error(f"Ошибка при создании таблиц: {e}")
                raise DatabaseError(f"Не удалось создать таблицы: {e}")
    
    async def add_sticker(
        self,
        user_id: int,
        prompt: str,
        file_path: str,
        username: Optional[str] = None,
        file_size: Optional[int] = None,
        generation_time: Optional[float] = None,
        model_version: Optional[str] = None
    ) -> str:
        """
        Добавляет информацию о сгенерированном стикере.
        
        Args:
            user_id: ID пользователя Telegram
            prompt: Текстовое описание для генерации
            file_path: Путь к файлу стикера
            username: Имя пользователя (опционально)
            file_size: Размер файла в байтах (опционально)
            generation_time: Время генерации в секундах (опционально)
            model_version: Версия использованной модели (опционально)
            
        Returns:
            ID добавленного стикера
        """
        # Генерируем уникальный ID
        sticker_id = str(uuid.uuid4())
        
        async with self._get_connection() as conn:
            try:
                # Добавляем стикер
                await conn.execute("""
                    INSERT INTO stickers 
                    (id, user_id, username, prompt, file_path, file_size, 
                     generation_time, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sticker_id, user_id, username, prompt, file_path,
                    file_size, generation_time, model_version
                ))
                
                # Обновляем статистику пользователя
                await conn.execute("""
                    INSERT INTO user_stats (user_id, username, total_stickers, last_use)
                    VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username = COALESCE(excluded.username, user_stats.username),
                        total_stickers = user_stats.total_stickers + 1,
                        last_use = CURRENT_TIMESTAMP
                """, (user_id, username))
                
                await conn.commit()
                logger.info(
                    f"Стикер {sticker_id} добавлен для пользователя "
                    f"{user_id} ({username or 'Unknown'})"
                )
                
                return sticker_id
                
            except Exception as e:
                await conn.rollback()
                logger.error(f"Ошибка при добавлении стикера: {e}")
                raise DatabaseError(f"Не удалось добавить стикер: {e}")
    
    async def update_user_request_stats(
        self,
        user_id: int,
        request_type: str,
        username: Optional[str] = None
    ):
        """
        Обновляет статистику запросов пользователя.
        
        Args:
            user_id: ID пользователя
            request_type: Тип запроса ('voice' или 'text')
            username: Имя пользователя (опционально)
        """
        if request_type not in ('voice', 'text'):
            raise ValueError(f"Неверный тип запроса: {request_type}")
        
        column = f"total_{request_type}_requests"
        
        async with self._get_connection() as conn:
            try:
                await conn.execute(f"""
                    INSERT INTO user_stats (user_id, username, {column}, last_use)
                    VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username = COALESCE(?, user_stats.username),
                        {column} = user_stats.{column} + 1,
                        last_use = CURRENT_TIMESTAMP
                """, (user_id, username, username))
                
                await conn.commit()
                
            except Exception as e:
                await conn.rollback()
                logger.error(f"Ошибка обновления статистики: {e}")
    
    async def log_error(
        self,
        user_id: int,
        error_type: str,
        error_message: str,
        prompt: Optional[str] = None
    ):
        """
        Сохраняет информацию об ошибке в БД.
        
        Args:
            user_id: ID пользователя
            error_type: Тип ошибки (transcription, generation, etc.)
            error_message: Сообщение об ошибке
            prompt: Промт, вызвавший ошибку (опционально)
        """
        async with self._get_connection() as conn:
            try:
                await conn.execute("""
                    INSERT INTO error_log (user_id, error_type, error_message, prompt)
                    VALUES (?, ?, ?, ?)
                """, (user_id, error_type, error_message, prompt))
                
                await conn.commit()
                
            except Exception as e:
                # Не прерываем работу из-за ошибки логирования
                logger.error(f"Не удалось сохранить ошибку в БД: {e}")
    
    async def get_user_stickers(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Получает стикеры пользователя.
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество записей
            offset: Смещение для пагинации
            
        Returns:
            Список словарей с информацией о стикерах
        """
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute("""
                    SELECT id, prompt, file_path, file_size, 
                           generation_time, model_version, created_at
                    FROM stickers 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (user_id, limit, offset))
                
                rows = await cursor.fetchall()
                
                stickers = []
                for row in rows:
                    stickers.append({
                        'id': row[0],
                        'prompt': row[1],
                        'file_path': row[2],
                        'file_size': row[3],
                        'generation_time': row[4],
                        'model_version': row[5],
                        'created_at': row[6]
                    })
                
                logger.info(f"Получено {len(stickers)} стикеров для пользователя {user_id}")
                return stickers
                
            except Exception as e:
                logger.error(f"Ошибка получения стикеров: {e}")
                return []
    
    async def get_user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает статистику пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь со статистикой или None
        """
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute("""
                    SELECT username, total_stickers, total_voice_requests,
                           total_text_requests, first_use, last_use
                    FROM user_stats
                    WHERE user_id = ?
                """, (user_id,))
                
                row = await cursor.fetchone()
                
                if row:
                    return {
                        'username': row[0],
                        'total_stickers': row[1],
                        'total_voice_requests': row[2],
                        'total_text_requests': row[3],
                        'first_use': row[4],
                        'last_use': row[5]
                    }
                
                return None
                
            except Exception as e:
                logger.error(f"Ошибка получения статистики: {e}")
                return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Получает общую статистику базы данных.
        
        Returns:
            Словарь с общей статистикой
        """
        async with self._get_connection() as conn:
            try:
                stats = {}
                
                # Общее количество стикеров
                cursor = await conn.execute("SELECT COUNT(*) FROM stickers")
                stats['total_stickers'] = (await cursor.fetchone())[0]
                
                # Количество уникальных пользователей
                cursor = await conn.execute("SELECT COUNT(DISTINCT user_id) FROM stickers")
                stats['unique_users'] = (await cursor.fetchone())[0]
                
                # Статистика за последние 24 часа
                cursor = await conn.execute("""
                    SELECT COUNT(*) FROM stickers 
                    WHERE created_at > datetime('now', '-1 day')
                """)
                stats['stickers_24h'] = (await cursor.fetchone())[0]
                
                # Самые популярные промпты
                cursor = await conn.execute("""
                    SELECT prompt, COUNT(*) as count 
                    FROM stickers 
                    GROUP BY prompt 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                stats['top_prompts'] = await cursor.fetchall()
                
                return stats
                
            except Exception as e:
                logger.error(f"Ошибка получения статистики: {e}")
                return {}
    
    async def cleanup_old_errors(self, days: int = 30):
        """
        Удаляет старые записи об ошибках.
        
        Args:
            days: Количество дней для хранения ошибок
        """
        async with self._get_connection() as conn:
            try:
                await conn.execute("""
                    DELETE FROM error_log 
                    WHERE timestamp < datetime('now', ? || ' days')
                """, (-days,))
                
                await conn.commit()
                
                logger.info(f"Старые ошибки (старше {days} дней) удалены")
                
            except Exception as e:
                logger.error(f"Ошибка при очистке старых ошибок: {e}")
    
    async def close_connection(self):
        """Закрывает соединение с базой данных."""
        if self._connection:
            try:
                await self._connection.close()
                self._connection = None
                logger.info("Соединение с БД закрыто")
            except Exception as e:
                logger.error(f"Ошибка при закрытии соединения: {e}")
    
    async def __aenter__(self):
        """Вход в контекстный менеджер."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера."""
        await self.close_connection()