"""
Advanced database manager for SQLite with async support, migrations, and caching.
Provides robust data persistence with type safety and performance optimizations.
"""
import aiosqlite
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union, TypeVar, Generic, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import json
import hashlib
from abc import ABC, abstractmethod
import time

from logger import get_logger, log_execution_time, log_error
from config import config

logger = get_logger(__name__)

T = TypeVar('T')


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class ConnectionError(DatabaseError):
    """Database connection error."""
    pass


class QueryError(DatabaseError):
    """Query execution error."""
    pass


class IntegrityError(DatabaseError):
    """Data integrity error."""
    pass


@dataclass
class Sticker:
    """Sticker entity model."""
    id: str
    user_id: int
    username: Optional[str]
    prompt: str
    file_path: str
    file_size: Optional[int]
    generation_time: Optional[float]
    model_version: Optional[str]
    created_at: datetime
    style: Optional[str] = None
    background: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> 'Sticker':
        """Create Sticker from database row."""
        data = dict(row)
        # Parse datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        # Parse metadata JSON
        if data.get('metadata') and isinstance(data['metadata'], str):
            data['metadata'] = json.loads(data['metadata'])
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        # Convert datetime to string
        if isinstance(data.get('created_at'), datetime):
            data['created_at'] = data['created_at'].isoformat()
        # Convert metadata to JSON
        if data.get('metadata'):
            data['metadata'] = json.dumps(data['metadata'])
        return data


@dataclass
class UserStats:
    """User statistics model."""
    user_id: int
    username: Optional[str]
    total_stickers: int
    total_voice_requests: int
    total_text_requests: int
    first_use: datetime
    last_use: datetime
    total_generation_time: float = 0.0
    favorite_style: Optional[str] = None
    favorite_background: Optional[str] = None
    
    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> 'UserStats':
        """Create UserStats from database row."""
        data = dict(row)
        # Parse datetimes
        for field_name in ['first_use', 'last_use']:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**data)


@dataclass
class ErrorLog:
    """Error log entry model."""
    id: Optional[int]
    user_id: int
    error_type: str
    error_message: str
    prompt: Optional[str]
    timestamp: datetime
    traceback: Optional[str] = None
    resolved: bool = False
    
    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> 'ErrorLog':
        """Create ErrorLog from database row."""
        data = dict(row)
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['resolved'] = bool(data.get('resolved', 0))
        return cls(**data)


class CacheManager:
    """Simple in-memory cache for database queries."""
    
    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache with TTL in seconds."""
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl_seconds
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        """Set value in cache."""
        async with self._lock:
            self.cache[key] = (value, time.time())
    
    async def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern."""
        async with self._lock:
            if pattern:
                keys_to_delete = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.cache[key]
            else:
                self.cache.clear()
    
    async def cleanup(self):
        """Remove expired entries."""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                k for k, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]


class QueryBuilder:
    """SQL query builder for type safety."""
    
    @staticmethod
    def insert(table: str, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build INSERT query."""
        columns = list(data.keys())
        placeholders = ['?' for _ in columns]
        values = list(data.values())
        
        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        return query, values
    
    @staticmethod
    def update(table: str, data: Dict[str, Any], where: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build UPDATE query."""
        set_clauses = [f"{col} = ?" for col in data.keys()]
        where_clauses = [f"{col} = ?" for col in where.keys()]
        values = list(data.values()) + list(where.values())
        
        query = f"""
            UPDATE {table}
            SET {', '.join(set_clauses)}
            WHERE {' AND '.join(where_clauses)}
        """
        return query, values
    
    @staticmethod
    def select(
        table: str,
        columns: List[str] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Tuple[str, List[Any]]:
        """Build SELECT query."""
        columns_str = ', '.join(columns) if columns else '*'
        query_parts = [f"SELECT {columns_str} FROM {table}"]
        values = []
        
        if where:
            where_clauses = [f"{col} = ?" for col in where.keys()]
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
            values.extend(where.values())
        
        if order_by:
            query_parts.append(f"ORDER BY {order_by}")
        
        if limit:
            query_parts.append(f"LIMIT {limit}")
        
        if offset:
            query_parts.append(f"OFFSET {offset}")
        
        return ' '.join(query_parts), values


class DatabaseMigration:
    """Database migration system."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.migrations: List[Callable] = []
    
    def register(self, version: int):
        """Decorator to register migration."""
        def decorator(func: Callable):
            self.migrations.append((version, func))
            return func
        return decorator
    
    async def run_migrations(self, conn: aiosqlite.Connection):
        """Run pending migrations."""
        # Create migrations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Get current version
        cursor = await conn.execute(
            "SELECT MAX(version) FROM schema_migrations"
        )
        current_version = (await cursor.fetchone())[0] or 0
        
        # Sort migrations by version
        self.migrations.sort(key=lambda x: x[0])
        
        # Run pending migrations
        for version, migration in self.migrations:
            if version > current_version:
                logger.info(f"Running migration {version}")
                await migration(conn)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)",
                    (version,)
                )
                await conn.commit()
                logger.info(f"Migration {version} completed")


class DatabasePool:
    """Connection pool for better concurrency."""
    
    def __init__(self, db_path: Path, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize connection pool."""
        async with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                await self.connections.put(conn)
            
            self._initialized = True
            logger.info(f"Database pool initialized with {self.pool_size} connections")
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection."""
        conn = await aiosqlite.connect(
            self.db_path,
            timeout=config.performance.db_timeout_sec
        )
        # Enable optimizations
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")
        await conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        await conn.execute("PRAGMA temp_store = MEMORY")
        
        # Set row factory
        conn.row_factory = aiosqlite.Row
        
        return conn
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        if not self._initialized:
            await self.initialize()
        
        conn = await self.connections.get()
        try:
            yield conn
        finally:
            await self.connections.put(conn)
    
    async def close(self):
        """Close all connections in pool."""
        async with self._lock:
            while not self.connections.empty():
                conn = await self.connections.get()
                await conn.close()
            self._initialized = False


class DatabaseManager:
    """Enhanced database manager with advanced features."""
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize database manager."""
        self.db_path = Path(db_path)
        self.pool = DatabasePool(self.db_path, pool_size=config.performance.max_workers)
        self.cache = CacheManager(ttl_seconds=3600)  # 1 hour cache
        self.migrations = DatabaseMigration(self.db_path)
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Register migrations
        self._register_migrations()
    
    def _register_migrations(self):
        """Register database migrations."""
        
        @self.migrations.register(1)
        async def initial_schema(conn: aiosqlite.Connection):
            """Initial database schema."""
            # Stickers table
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
                    style TEXT,
                    background TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    CHECK (user_id > 0),
                    CHECK (length(prompt) > 0),
                    CHECK (file_size >= 0)
                )
            """)
            
            # Indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON stickers(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON stickers(created_at DESC)")
        #             await conn.execute("ALTER TABLE stickers ADD COLUMN style TEXT DEFAULT 'cartoon'")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_style ON stickers(style)")
            
            # User stats table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    total_stickers INTEGER DEFAULT 0,
                    total_voice_requests INTEGER DEFAULT 0,
                    total_text_requests INTEGER DEFAULT 0,
                    total_generation_time REAL DEFAULT 0.0,
                    favorite_style TEXT,
                    favorite_background TEXT,
                    first_use TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_use TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    CHECK (total_stickers >= 0),
                    CHECK (total_voice_requests >= 0),
                    CHECK (total_text_requests >= 0)
                )
            """)
            
            # Error log table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    prompt TEXT,
                    traceback TEXT,
                    resolved INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_error_timestamp ON error_log(timestamp DESC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_error_user ON error_log(user_id)")
        
        @self.migrations.register(2)
        async def add_analytics_tables(conn: aiosqlite.Connection):
            """Add analytics and metrics tables."""
            # Daily statistics
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date DATE PRIMARY KEY,
                    total_users INTEGER DEFAULT 0,
                    total_stickers INTEGER DEFAULT 0,
                    total_voice_requests INTEGER DEFAULT 0,
                    total_text_requests INTEGER DEFAULT 0,
                    total_errors INTEGER DEFAULT 0,
                    avg_generation_time REAL DEFAULT 0.0
                )
            """)
            
            # Generation metrics
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    prompt_length INTEGER,
                    generation_time REAL,
                    model_version TEXT,
                    style TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        @self.migrations.register(3)
        async def add_user_preferences(conn: aiosqlite.Connection):
            """Add user preferences table."""
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER PRIMARY KEY,
                    language TEXT DEFAULT 'ru',
                    default_style TEXT,
                    default_background TEXT,
                    enable_notifications INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def initialize(self):
        """Initialize database and run migrations."""
        logger.info("Initializing database...")
        
        # Initialize connection pool
        await self.pool.initialize()
        
        # Run migrations
        async with self.pool.acquire() as conn:
            await self.migrations.run_migrations(conn)
            await conn.commit()
        
        logger.info("Database initialized successfully")
    
    async def close(self):
        """Close database connections."""
        await self.pool.close()
        logger.info("Database connections closed")
    
    # === Sticker Operations ===
    
    @log_execution_time
    @log_error("database")
    async def add_sticker(
        self,
        user_id: int,
        prompt: str,
        file_path: str,
        username: Optional[str] = None,
        file_size: Optional[int] = None,
        generation_time: Optional[float] = None,
        model_version: Optional[str] = None,
        style: Optional[str] = None,
        background: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new sticker to database."""
        sticker_id = str(uuid.uuid4())
        
        sticker = Sticker(
            id=sticker_id,
            user_id=user_id,
            username=username,
            prompt=prompt,
            file_path=file_path,
            file_size=file_size,
            generation_time=generation_time,
            model_version=model_version,
            style=style,
            background=background,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        
        async with self.pool.acquire() as conn:
            # Insert sticker
            query, values = QueryBuilder.insert('stickers', sticker.to_dict())
            await conn.execute(query, values)
            
            # Update user stats
            await self._update_user_stats(
                conn, user_id, username,
                increment_stickers=True,
                add_generation_time=generation_time,
                style=style,
                background=background
            )
            
            # Update daily stats
            await self._update_daily_stats(
                conn,
                increment_stickers=True,
                add_generation_time=generation_time
            )
            
            # Log generation metrics
            if generation_time:
                await conn.execute("""
                    INSERT INTO generation_metrics 
                    (user_id, prompt_length, generation_time, model_version, style)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, len(prompt), generation_time, model_version, style))
            
            await conn.commit()
        
        # Invalidate cache
        await self.cache.invalidate(f"user_stickers:{user_id}")
        await self.cache.invalidate(f"user_stats:{user_id}")
        
        logger.info(f"Sticker {sticker_id} added for user {user_id}")
        return sticker_id
    
    async def get_sticker(self, sticker_id: str) -> Optional[Sticker]:
        """Get sticker by ID."""
        # Check cache
        cache_key = f"sticker:{sticker_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        async with self.pool.acquire() as conn:
            query, values = QueryBuilder.select(
                'stickers',
                where={'id': sticker_id}
            )
            cursor = await conn.execute(query, values)
            row = await cursor.fetchone()
            
            if row:
                sticker = Sticker.from_row(row)
                await self.cache.set(cache_key, sticker)
                return sticker
            
            return None
    
    async def get_user_stickers(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
        style: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Sticker]:
        """Get user's stickers with filtering."""
        # Build cache key
        cache_key = f"user_stickers:{user_id}:{limit}:{offset}:{style}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        async with self.pool.acquire() as conn:
            # Build query with filters
            query_parts = ["SELECT * FROM stickers WHERE user_id = ?"]
            values = [user_id]
            
            if style:
                query_parts.append("AND style = ?")
                values.append(style)
            
            if date_from:
                query_parts.append("AND created_at >= ?")
                values.append(date_from.isoformat())
            
            if date_to:
                query_parts.append("AND created_at <= ?")
                values.append(date_to.isoformat())
            
            query_parts.append("ORDER BY created_at DESC")
            query_parts.append(f"LIMIT {limit} OFFSET {offset}")
            
            query = " ".join(query_parts)
            cursor = await conn.execute(query, values)
            rows = await cursor.fetchall()
            
            stickers = [Sticker.from_row(row) for row in rows]
            
            # Cache results
            await self.cache.set(cache_key, stickers)
            
            return stickers
    
    async def delete_sticker(self, sticker_id: str) -> bool:
        """Delete a sticker."""
        async with self.pool.acquire() as conn:
            # Get sticker info first
            sticker = await self.get_sticker(sticker_id)
            if not sticker:
                return False
            
            # Delete sticker
            await conn.execute("DELETE FROM stickers WHERE id = ?", (sticker_id,))
            
            # Update user stats
            await conn.execute("""
                UPDATE user_stats 
                SET total_stickers = total_stickers - 1
                WHERE user_id = ?
            """, (sticker.user_id,))
            
            await conn.commit()
        
        # Invalidate cache
        await self.cache.invalidate(f"sticker:{sticker_id}")
        await self.cache.invalidate(f"user_stickers:{sticker.user_id}")
        
        return True
    
    # === User Operations ===
    
    async def update_user_request_stats(
        self,
        user_id: int,
        request_type: str,
        username: Optional[str] = None
    ):
        """Update user request statistics."""
        if request_type not in ('voice', 'text'):
            raise ValueError(f"Invalid request type: {request_type}")
        
        column = f"total_{request_type}_requests"
        
        async with self.pool.acquire() as conn:
            await self._update_user_stats(
                conn, user_id, username,
                increment_voice=request_type == 'voice',
                increment_text=request_type == 'text'
            )
            
            await self._update_daily_stats(
                conn,
                increment_voice=request_type == 'voice',
                increment_text=request_type == 'text'
            )
            
            await conn.commit()
        
        # Invalidate cache
        await self.cache.invalidate(f"user_stats:{user_id}")
    
    async def get_user_stats(self, user_id: int) -> Optional[UserStats]:
        """Get user statistics."""
        # Check cache
        cache_key = f"user_stats:{user_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        async with self.pool.acquire() as conn:
            cursor = await conn.execute(
                "SELECT * FROM user_stats WHERE user_id = ?",
                (user_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                stats = UserStats.from_row(row)
                await self.cache.set(cache_key, stats)
                return stats
            
            return None
    
    async def get_user_preferences(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user preferences."""
        async with self.pool.acquire() as conn:
            cursor = await conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?",
                (user_id,)
            )
            row = await cursor.fetchone()
            
            return dict(row) if row else None
    
    async def update_user_preferences(
        self,
        user_id: int,
        **preferences
    ):
        """Update user preferences."""
        async with self.pool.acquire() as conn:
            # Check if preferences exist
            cursor = await conn.execute(
                "SELECT 1 FROM user_preferences WHERE user_id = ?",
                (user_id,)
            )
            exists = await cursor.fetchone()
            
            preferences['updated_at'] = datetime.utcnow().isoformat()
            
            if exists:
                # Update existing
                query, values = QueryBuilder.update(
                    'user_preferences',
                    preferences,
                    {'user_id': user_id}
                )
            else:
                # Insert new
                preferences['user_id'] = user_id
                query, values = QueryBuilder.insert(
                    'user_preferences',
                    preferences
                )
            
            await conn.execute(query, values)
            await conn.commit()
    
    # === Error Logging ===
    
    async def log_error(
        self,
        user_id: int,
        error_type: str,
        error_message: str,
        prompt: Optional[str] = None,
        traceback: Optional[str] = None
    ):
        """Log an error to database."""
        error_log = ErrorLog(
            id=None,
            user_id=user_id,
            error_type=error_type,
            error_message=error_message,
            prompt=prompt,
            traceback=traceback,
            timestamp=datetime.utcnow(),
            resolved=False
        )
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO error_log 
                (user_id, error_type, error_message, prompt, traceback)
                VALUES (?, ?, ?, ?, ?)
            """, (
                error_log.user_id,
                error_log.error_type,
                error_log.error_message,
                error_log.prompt,
                error_log.traceback
            ))
            
            # Update daily stats
            await self._update_daily_stats(conn, increment_errors=True)
            
            await conn.commit()
    
    async def get_recent_errors(
        self,
        limit: int = 100,
        unresolved_only: bool = True
    ) -> List[ErrorLog]:
        """Get recent errors."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM error_log"
            if unresolved_only:
                query += " WHERE resolved = 0"
            query += " ORDER BY timestamp DESC LIMIT ?"
            
            cursor = await conn.execute(query, (limit,))
            rows = await cursor.fetchall()
            
            return [ErrorLog.from_row(row) for row in rows]
    
    # === Analytics ===
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        async with self.pool.acquire() as conn:
            stats = {}
            
            # Total counts
            for table, field in [
                ('stickers', 'total_stickers'),
                ('user_stats', 'total_users')
            ]:
                cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[field] = (await cursor.fetchone())[0]
            
            # Recent activity (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM stickers 
                WHERE created_at > ?
            """, (yesterday.isoformat(),))
            stats['stickers_24h'] = (await cursor.fetchone())[0]
            
            # Popular styles
            cursor = await conn.execute("""
                SELECT style, COUNT(*) as count 
                FROM stickers 
                WHERE style IS NOT NULL
                GROUP BY style 
                ORDER BY count DESC 
                LIMIT 5
            """)
            stats['popular_styles'] = [
                {'style': row['style'], 'count': row['count']}
                for row in await cursor.fetchall()
            ]
            
            # Average generation time
            cursor = await conn.execute("""
                SELECT AVG(generation_time) 
                FROM stickers 
                WHERE generation_time IS NOT NULL
            """)
            stats['avg_generation_time'] = (await cursor.fetchone())[0] or 0
            
            # Error rate (last 24 hours)
            cursor = await conn.execute("""
                SELECT 
                    COUNT(CASE WHEN error_type IS NOT NULL THEN 1 END) as errors,
                    COUNT(*) as total
                FROM (
                    SELECT 'error' as error_type FROM error_log WHERE timestamp > ?
                    UNION ALL
                    SELECT NULL FROM stickers WHERE created_at > ?
                )
            """, (yesterday.isoformat(), yesterday.isoformat()))
            row = await cursor.fetchone()
            stats['error_rate_24h'] = (row['errors'] / row['total'] * 100) if row['total'] > 0 else 0
            
            # User activity distribution
            cursor = await conn.execute("""
                SELECT 
                    CASE 
                        WHEN total_stickers = 0 THEN 'inactive'
                        WHEN total_stickers < 5 THEN 'low'
                        WHEN total_stickers < 20 THEN 'medium'
                        ELSE 'high'
                    END as activity_level,
                    COUNT(*) as count
                FROM user_stats
                GROUP BY activity_level
            """)
            stats['user_activity'] = {
                row['activity_level']: row['count']
                for row in await cursor.fetchall()
            }
            
            return stats
    
    async def get_daily_stats(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get daily statistics."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM daily_stats"
            values = []
            
            if date_from or date_to:
                where_clauses = []
                if date_from:
                    where_clauses.append("date >= ?")
                    values.append(date_from.date().isoformat())
                if date_to:
                    where_clauses.append("date <= ?")
                    values.append(date_to.date().isoformat())
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY date DESC"
            
            cursor = await conn.execute(query, values)
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # === Maintenance ===
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self.pool.acquire() as conn:
            # Delete old errors
            deleted_errors = await conn.execute("""
                DELETE FROM error_log 
                WHERE timestamp < ? AND resolved = 1
            """, (cutoff_date.isoformat(),))
            
            # Delete old metrics
            deleted_metrics = await conn.execute("""
                DELETE FROM generation_metrics 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            await conn.commit()
            
            logger.info(
                f"Cleanup completed: {deleted_errors.rowcount} errors, "
                f"{deleted_metrics.rowcount} metrics deleted"
            )
    
    async def vacuum(self):
        """Vacuum database to reclaim space."""
        async with self.pool.acquire() as conn:
            await conn.execute("VACUUM")
            await conn.execute("ANALYZE")
            logger.info("Database vacuumed and analyzed")
    
    # === Private Helper Methods ===
    
    async def _update_user_stats(
        self,
        conn: aiosqlite.Connection,
        user_id: int,
        username: Optional[str] = None,
        increment_stickers: bool = False,
        increment_voice: bool = False,
        increment_text: bool = False,
        add_generation_time: Optional[float] = None,
        style: Optional[str] = None,
        background: Optional[str] = None
    ):
        """Update user statistics."""
        # Check if user exists
        cursor = await conn.execute(
            "SELECT 1 FROM user_stats WHERE user_id = ?",
            (user_id,)
        )
        exists = await cursor.fetchone()
        
        if exists:
            # Build update query
            updates = ["last_use = CURRENT_TIMESTAMP"]
            values = []
            
            if username:
                updates.append("username = ?")
                values.append(username)
            
            if increment_stickers:
                updates.append("total_stickers = total_stickers + 1")
            
            if increment_voice:
                updates.append("total_voice_requests = total_voice_requests + 1")
            
            if increment_text:
                updates.append("total_text_requests = total_text_requests + 1")
            
            if add_generation_time:
                updates.append("total_generation_time = total_generation_time + ?")
                values.append(add_generation_time)
            
            # Update favorite style/background based on frequency
            if style:
                # This would need more complex logic to track frequency
                # For now, just update to latest
                updates.append("favorite_style = ?")
                values.append(style)
            
            if background:
                updates.append("favorite_background = ?")
                values.append(background)
            
            values.append(user_id)
            
            await conn.execute(
                f"UPDATE user_stats SET {', '.join(updates)} WHERE user_id = ?",
                values
            )
        else:
            # Insert new user
            await conn.execute("""
                INSERT INTO user_stats 
                (user_id, username, total_stickers, total_voice_requests, 
                 total_text_requests, total_generation_time, favorite_style, 
                 favorite_background)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                username,
                1 if increment_stickers else 0,
                1 if increment_voice else 0,
                1 if increment_text else 0,
                add_generation_time or 0.0,
                style,
                background
            ))
    
    async def _update_daily_stats(
        self,
        conn: aiosqlite.Connection,
        increment_stickers: bool = False,
        increment_voice: bool = False,
        increment_text: bool = False,
        increment_errors: bool = False,
        add_generation_time: Optional[float] = None
    ):
        """Update daily statistics."""
        today = datetime.utcnow().date().isoformat()
        
        # Check if today's record exists
        cursor = await conn.execute(
            "SELECT 1 FROM daily_stats WHERE date = ?",
            (today,)
        )
        exists = await cursor.fetchone()
        
        if exists:
            # Build update query
            updates = []
            values = []
            
            if increment_stickers:
                updates.append("total_stickers = total_stickers + 1")
            
            if increment_voice:
                updates.append("total_voice_requests = total_voice_requests + 1")
            
            if increment_text:
                updates.append("total_text_requests = total_text_requests + 1")
            
            if increment_errors:
                updates.append("total_errors = total_errors + 1")
            
            if add_generation_time:
                # Update average generation time
                # This is simplified - in production, you'd track count and sum
                updates.append("avg_generation_time = ?")
                values.append(add_generation_time)
            
            if updates:
                values.append(today)
                await conn.execute(
                    f"UPDATE daily_stats SET {', '.join(updates)} WHERE date = ?",
                    values
                )
        else:
            # Insert new record
            await conn.execute("""
                INSERT INTO daily_stats 
                (date, total_stickers, total_voice_requests, total_text_requests, 
                 total_errors, avg_generation_time, total_users)
                VALUES (?, ?, ?, ?, ?, ?, 
                    (SELECT COUNT(DISTINCT user_id) FROM stickers WHERE date(created_at) = ?))
            """, (
                today,
                1 if increment_stickers else 0,
                1 if increment_voice else 0,
                1 if increment_text else 0,
                1 if increment_errors else 0,
                add_generation_time or 0.0,
                today
            ))
    
    # === Context Manager Support ===
    
    async def __aenter__(self):
        """Enter async context."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.close()