"""
Advanced bot handlers for Telegram bot "Golostickery".
Implements comprehensive user interaction with rate limiting, queuing, and analytics.
"""
import os
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import json

from aiogram import Dispatcher, types, F, Bot
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.types import (
    ContentType, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton,
    CallbackQuery, Message, ReplyKeyboardMarkup, KeyboardButton,
    ReplyKeyboardRemove, InputMediaPhoto
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramBadRequest, TelegramAPIError
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram.filters.callback_data import CallbackData

from stt_service.service import transcribe_audio, TranscriptionError
from image_generation_service.service import (
    generate_sticker_image, GenerationError, 
    get_available_styles, estimate_generation_time
)
from sticker_utils.utils import (
    process_image_for_sticker, save_image_to_file,
    BackgroundStyle, cleanup_old_files, get_image_info
)
from db_manager.manager import DatabaseManager, DatabaseError
from logger import get_logger, log_context, log_execution_time
from config import config

logger = get_logger(__name__)


# === States ===

class StickerGeneration(StatesGroup):
    """States for sticker generation flow."""
    waiting_for_prompt = State()
    waiting_for_style = State()
    waiting_for_background = State()
    processing = State()
    
    # Advanced states
    waiting_for_custom_style = State()
    waiting_for_feedback = State()
    editing_prompt = State()


class UserSettings(StatesGroup):
    """States for user settings."""
    main_menu = State()
    changing_language = State()
    changing_default_style = State()
    changing_default_background = State()


# === Callback Data ===

class StyleCallback(CallbackData, prefix="style"):
    """Callback data for style selection."""
    style_id: str
    action: str = "select"  # select, info, preview


class BackgroundCallback(CallbackData, prefix="bg"):
    """Callback data for background selection."""
    background_id: str
    action: str = "select"


class NavigationCallback(CallbackData, prefix="nav"):
    """Callback data for navigation."""
    action: str  # back, cancel, help, settings
    target: Optional[str] = None


class StickerCallback(CallbackData, prefix="sticker"):
    """Callback data for sticker actions."""
    sticker_id: str
    action: str  # regenerate, delete, share, save


# === Rate Limiting ===

class RateLimiter:
    """Rate limiter for user requests."""
    
    def __init__(self):
        self.user_requests: Dict[int, List[float]] = defaultdict(list)
        self.user_warnings: Dict[int, int] = defaultdict(int)
    
    def check_rate_limit(self, user_id: int) -> tuple[bool, Optional[int]]:
        """
        Check if user exceeded rate limit.
        Returns (is_allowed, seconds_to_wait)
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        # Clean old requests
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if req_time > hour_ago
        ]
        
        # Count requests
        minute_requests = sum(
            1 for req_time in self.user_requests[user_id]
            if req_time > minute_ago
        )
        hour_requests = len(self.user_requests[user_id])
        
        # Check limits
        if minute_requests >= config.security.rate_limit_messages_per_minute:
            wait_time = int(60 - (now - self.user_requests[user_id][-config.security.rate_limit_messages_per_minute]))
            return False, wait_time
        
        if hour_requests >= config.security.rate_limit_messages_per_hour:
            wait_time = int(3600 - (now - self.user_requests[user_id][-config.security.rate_limit_messages_per_hour]))
            return False, wait_time
        
        # Add request
        self.user_requests[user_id].append(now)
        return True, None
    
    def add_warning(self, user_id: int) -> int:
        """Add warning for user and return warning count."""
        self.user_warnings[user_id] += 1
        return self.user_warnings[user_id]


# === Generation Queue ===

class GenerationQueue:
    """Queue system for managing generation requests."""
    
    def __init__(self, max_concurrent: int = 3):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing: Dict[int, str] = {}  # user_id -> request_id
        self.max_concurrent = max_concurrent
        self.workers: List[asyncio.Task] = []
    
    async def start_workers(self):
        """Start queue workers."""
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        logger.info(f"Started {self.max_concurrent} generation workers")
    
    async def stop_workers(self):
        """Stop all workers."""
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Stopped all generation workers")
    
    async def add_request(self, request: Dict[str, Any]) -> int:
        """Add generation request to queue. Returns position."""
        await self.queue.put(request)
        return self.queue.qsize()
    
    def get_position(self, request_id: str) -> int:
        """Get position in queue for request."""
        position = 1
        for item in list(self.queue._queue):
            if item.get('request_id') == request_id:
                return position
            position += 1
        return 0
    
    async def _worker(self, worker_id: int):
        """Queue worker process."""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get request from queue
                request = await self.queue.get()
                
                # Process request
                user_id = request['user_id']
                self.processing[user_id] = request['request_id']
                
                try:
                    await self._process_request(request)
                finally:
                    if user_id in self.processing:
                        del self.processing[user_id]
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
    
    async def _process_request(self, request: Dict[str, Any]):
        """Process a generation request."""
        bot: Bot = request['bot']
        message: Message = request['message']
        state: FSMContext = request['state']
        prompt: str = request['prompt']
        style: str = request['style']
        background: str = request['background']
        db_manager: DatabaseManager = request['db_manager']
        
        user = message.from_user
        
        try:
            # Generate image
            with log_context(user_id=user.id, request_id=request['request_id']):
                logger.info(f"Processing generation request: {prompt[:50]}...")
                
                # Update progress
                progress_msg = await message.answer(
                    f"🎨 Генерация началась...\n"
                    f"📝 Промпт: _{prompt}_\n"
                    f"⏱ Примерное время: {request['estimated_time']:.0f} сек"
                )
                
                # Generate
                start_time = time.time()
                generated_image = await generate_sticker_image(
                    prompt=prompt,
                    style=style,
                    enhance_prompt=True
                )
                
                if not generated_image:
                    raise GenerationError("Failed to generate image")
                
                generation_time = time.time() - start_time
                
                # Process image
                await progress_msg.edit_text("🖼 Обработка изображения...")
                
                background_style = BackgroundStyle(background)
                processed_image = await process_image_for_sticker(
                    generated_image,
                    background_style=background_style,
                    add_shadow=background in ['transparent', 'circle'],
                    add_outline=style in ['cartoon', 'anime'],
                    remove_white_bg=background == 'transparent'
                )
                
                if not processed_image:
                    raise GenerationError("Failed to process image")
                
                # Save file
                sticker_filename = f"sticker_{user.id}_{uuid.uuid4().hex[:8]}.png"
                sticker_path = await save_image_to_file(
                    processed_image,
                    sticker_filename,
                    str(config.paths.storage_dir)
                )
                
                if not sticker_path:
                    raise GenerationError("Failed to save image")
                
                # Get file info
                image_info = await get_image_info(sticker_path)
                
                # Send sticker with inline keyboard
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="🔄 Регенерировать",
                            callback_data=StickerCallback(
                                sticker_id=request['request_id'],
                                action="regenerate"
                            ).pack()
                        ),
                        InlineKeyboardButton(
                            text="💾 Сохранить",
                            callback_data=StickerCallback(
                                sticker_id=request['request_id'],
                                action="save"
                            ).pack()
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            text="📤 Поделиться",
                            callback_data=StickerCallback(
                                sticker_id=request['request_id'],
                                action="share"
                            ).pack()
                        ),
                        InlineKeyboardButton(
                            text="🗑 Удалить",
                            callback_data=StickerCallback(
                                sticker_id=request['request_id'],
                                action="delete"
                            ).pack()
                        )
                    ]
                ])
                
                caption = (
                    f"✅ Стикер готов!\n\n"
                    f"📝 Промпт: _{prompt}_\n"
                    f"🎨 Стиль: {style}\n"
                    f"🖼 Фон: {background}\n"
                    f"📐 Размер: {image_info['width']}x{image_info['height']}\n"
                    f"📦 Файл: {image_info['size_kb']:.1f} КБ\n"
                    f"⏱ Время: {generation_time:.1f} сек"
                )
                
                await bot.send_photo(
                    chat_id=message.chat.id,
                    photo=FSInputFile(sticker_path),
                    caption=caption,
                    reply_markup=keyboard
                )
                
                # Save to database
                sticker_id = await db_manager.add_sticker(
                    user_id=user.id,
                    prompt=prompt,
                    file_path=sticker_path,
                    username=user.username,
                    file_size=int(image_info['size_kb'] * 1024),
                    generation_time=generation_time,
                    model_version=f"SD1.5-{style}",
                    style=style,
                    background=background,
                    metadata={
                        'request_id': request['request_id'],
                        'width': image_info['width'],
                        'height': image_info['height']
                    }
                )
                
                # Delete progress message
                await progress_msg.delete()
                
                # Clear state
                await state.clear()
                
                logger.info(f"Sticker generated successfully: {sticker_id}")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            
            # Send error message
            error_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🔄 Попробовать снова",
                        callback_data=NavigationCallback(action="retry").pack()
                    ),
                    InlineKeyboardButton(
                        text="❌ Отменить",
                        callback_data=NavigationCallback(action="cancel").pack()
                    )
                ]
            ])
            
            await message.answer(
                "❌ Не удалось создать стикер. Попробуйте другое описание или повторите позже.",
                reply_markup=error_keyboard
            )
            
            # Log error
            await db_manager.log_error(
                user_id=user.id,
                error_type='generation',
                error_message=str(e),
                prompt=prompt
            )
            
            # Clear state
            await state.clear()


# === Decorators ===

def rate_limit_check(func: Callable) -> Callable:
    """Decorator to check rate limits."""
    @wraps(func)
    async def wrapper(message_or_callback: Union[Message, CallbackQuery], *args, **kwargs):
        # Get user from message or callback
        if isinstance(message_or_callback, Message):
            user = message_or_callback.from_user
            reply_func = message_or_callback.reply
        else:
            user = message_or_callback.from_user
            reply_func = message_or_callback.message.answer
        
        # Check rate limit
        rate_limiter: RateLimiter = kwargs.get('rate_limiter') or data.get('rate_limiter')
        is_allowed, wait_time = rate_limiter.check_rate_limit(user.id)
        
        if not is_allowed:
            warnings = rate_limiter.add_warning(user.id)
            
            if warnings >= 3:
                # Block user
                await reply_func(
                    "🚫 Вы превысили лимит запросов многократно. "
                    "Доступ временно заблокирован."
                )
                return
            
            await reply_func(
                f"⏳ Слишком много запросов. "
                f"Подождите {wait_time} секунд.\n"
                f"Предупреждение {warnings}/3"
            )
            return
        
        # Call original function
        return await func(message_or_callback, *args, **kwargs)
    
    return wrapper


def check_user_access(func: Callable) -> Callable:
    """Decorator to check user access permissions."""
    @wraps(func)
    async def wrapper(message_or_callback: Union[Message, CallbackQuery], *args, **kwargs):
        # Get user
        if isinstance(message_or_callback, Message):
            user = message_or_callback.from_user
            reply_func = message_or_callback.reply
        else:
            user = message_or_callback.from_user
            reply_func = message_or_callback.message.answer
        
        # Check if user is blocked
        if config.security.blocked_user_ids and user.id in config.security.blocked_user_ids:
            await reply_func("🚫 Доступ запрещен.")
            return
        
        # Check if whitelist is enabled
        if config.security.allowed_user_ids and user.id not in config.security.allowed_user_ids:
            await reply_func("🔒 Бот доступен только для авторизованных пользователей.")
            return
        
        return await func(message_or_callback, *args, **kwargs)
    
    return wrapper


# === Helper Functions ===

def create_main_keyboard() -> ReplyKeyboardMarkup:
    """Create main reply keyboard."""
    builder = ReplyKeyboardBuilder()
    
    builder.row(
        KeyboardButton(text="🎤 Голосовой стикер"),
        KeyboardButton(text="✍️ Текстовый стикер")
    )
    builder.row(
        KeyboardButton(text="🎨 Мои стикеры"),
        KeyboardButton(text="📊 Статистика")
    )
    builder.row(
        KeyboardButton(text="⚙️ Настройки"),
        KeyboardButton(text="📖 Помощь")
    )
    
    return builder.as_markup(resize_keyboard=True)


def create_style_keyboard(user_style: Optional[str] = None) -> InlineKeyboardMarkup:
    """Create style selection keyboard."""
    builder = InlineKeyboardBuilder()
    styles = asyncio.run(get_available_styles())
    
    # Add style buttons (2 per row)
    for i in range(0, len(styles), 2):
        row_buttons = []
        for j in range(2):
            if i + j < len(styles):
                style = styles[i + j]
                text = style['name']
                if user_style == style['id']:
                    text = f"✅ {text}"
                
                row_buttons.append(
                    InlineKeyboardButton(
                        text=text,
                        callback_data=StyleCallback(
                            style_id=style['id'],
                            action="select"
                        ).pack()
                    )
                )
        builder.row(*row_buttons)
    
    # Add control buttons
    builder.row(
        InlineKeyboardButton(
            text="🎨 Свой стиль",
            callback_data=StyleCallback(
                style_id="custom",
                action="select"
            ).pack()
        )
    )
    builder.row(
        InlineKeyboardButton(
            text="◀️ Назад",
            callback_data=NavigationCallback(action="back").pack()
        ),
        InlineKeyboardButton(
            text="❌ Отмена",
            callback_data=NavigationCallback(action="cancel").pack()
        )
    )
    
    return builder.as_markup()


def create_background_keyboard(user_bg: Optional[str] = None) -> InlineKeyboardMarkup:
    """Create background selection keyboard."""
    builder = InlineKeyboardBuilder()
    
    backgrounds = [
        ("transparent", "🏻 Прозрачный"),
        ("white", "⬜ Белый"),
        ("gradient", "🌈 Градиент"),
        ("circle", "⭕ Круглый"),
    ]
    
    # Add background buttons (2 per row)
    for i in range(0, len(backgrounds), 2):
        row_buttons = []
        for j in range(2):
            if i + j < len(backgrounds):
                bg_id, bg_name = backgrounds[i + j]
                text = bg_name
                if user_bg == bg_id:
                    text = f"✅ {text}"
                
                row_buttons.append(
                    InlineKeyboardButton(
                        text=text,
                        callback_data=BackgroundCallback(
                            background_id=bg_id,
                            action="select"
                        ).pack()
                    )
                )
        builder.row(*row_buttons)
    
    # Add control buttons
    builder.row(
        InlineKeyboardButton(
            text="◀️ Назад",
            callback_data=NavigationCallback(action="back", target="style").pack()
        ),
        InlineKeyboardButton(
            text="❌ Отмена",
            callback_data=NavigationCallback(action="cancel").pack()
        )
    )
    
    return builder.as_markup()


# === Command Handlers ===

async def start_command_handler(message: Message, state: FSMContext, db_manager: DatabaseManager):
    """Handle /start command."""
    user = message.from_user
    user_name = user.first_name or "Пользователь"
    
    with log_context(user_id=user.id, username=user.username):
        logger.info("Start command received")
        
        # Update user stats
        await db_manager.update_user_request_stats(user.id, 'text', user.username)
        
        # Get user preferences
        preferences = await db_manager.get_user_preferences(user.id)
        language = preferences.get('language', 'ru') if preferences else 'ru'
        
        # Send welcome message with main keyboard
        welcome_text = config.messages.get('start', language, name=user_name)
        
        await message.answer(
            welcome_text,
            reply_markup=create_main_keyboard()
        )
        
        # Show quick start guide for new users
        user_stats = await db_manager.get_user_stats(user.id)
        if not user_stats or user_stats.total_stickers == 0:
            await message.answer(
                "💡 *Быстрый старт:*\n\n"
                "1. Нажмите «🎤 Голосовой стикер» и запишите описание\n"
                "2. Или нажмите «✍️ Текстовый стикер» и напишите текст\n"
                "3. Выберите стиль и фон\n"
                "4. Получите уникальный стикер!\n\n"
                "Попробуйте прямо сейчас! 👇"
            )
        
        # Clear any existing state
        await state.clear()


async def help_command_handler(message: Message, db_manager: DatabaseManager):
    """Handle /help command."""
    user = message.from_user
    
    with log_context(user_id=user.id):
        logger.info("Help command received")
        
        # Get available styles
        styles = await get_available_styles()
        styles_text = "\n".join([f"   • {s['name']}: {s['description']}" for s in styles])
        
        help_text = config.messages.get('help') + f"\n\n🎨 *Доступные стили:*\n{styles_text}"
        
        # Create inline keyboard with useful links
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📹 Видео-инструкция",
                    url="https://example.com/tutorial"
                ),
                InlineKeyboardButton(
                    text="💬 Поддержка",
                    url="https://t.me/support"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🎨 Галерея примеров",
                    callback_data=NavigationCallback(action="gallery").pack()
                )
            ]
        ])
        
        await message.answer(help_text, reply_markup=keyboard)


@log_execution_time
async def stats_command_handler(message: Message, db_manager: DatabaseManager):
    """Handle /stats command with detailed statistics."""
    user = message.from_user
    
    with log_context(user_id=user.id):
        logger.info("Stats command received")
        
        try:
            # Get user statistics
            user_stats = await db_manager.get_user_stats(user.id)
            
            if not user_stats:
                await message.answer(
                    "📊 У вас пока нет статистики.\n"
                    "Создайте свой первый стикер! 🎨"
                )
                return
            
            # Get additional stats
            recent_stickers = await db_manager.get_user_stickers(user.id, limit=5)
            
            # Format statistics
            stats_text = (
                "📊 *Ваша статистика:*\n\n"
                f"🎨 Стикеров создано: {user_stats.total_stickers}\n"
                f"🎤 Голосовых запросов: {user_stats.total_voice_requests}\n"
                f"✍️ Текстовых запросов: {user_stats.total_text_requests}\n"
                f"⏱ Общее время генерации: {user_stats.total_generation_time:.1f} сек\n"
                f"📅 Первое использование: {user_stats.first_use.strftime('%d.%m.%Y')}\n"
                f"🕐 Последнее использование: {user_stats.last_use.strftime('%d.%m.%Y %H:%M')}\n"
            )
            
            if user_stats.favorite_style:
                stats_text += f"❤️ Любимый стиль: {user_stats.favorite_style}\n"
            
            if user_stats.favorite_background:
                stats_text += f"🖼 Любимый фон: {user_stats.favorite_background}\n"
            
            # Add achievements
            achievements = []
            if user_stats.total_stickers >= 100:
                achievements.append("🏆 Мастер стикеров (100+)")
            elif user_stats.total_stickers >= 50:
                achievements.append("🥇 Опытный создатель (50+)")
            elif user_stats.total_stickers >= 10:
                achievements.append("🥈 Активный пользователь (10+)")
            
            if achievements:
                stats_text += f"\n🎖 *Достижения:*\n" + "\n".join(achievements)
            
            # Create keyboard
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📈 Подробная аналитика",
                        callback_data=NavigationCallback(action="detailed_stats").pack()
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="🎨 Мои стикеры",
                        callback_data=NavigationCallback(action="my_stickers").pack()
                    ),
                    InlineKeyboardButton(
                        text="🏆 Лидерборд",
                        callback_data=NavigationCallback(action="leaderboard").pack()
                    )
                ]
            ])
            
            await message.answer(stats_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            await message.answer("❌ Не удалось получить статистику. Попробуйте позже.")


async def mystickers_command_handler(message: Message, db_manager: DatabaseManager):
    """Handle /mystickers command with pagination."""
    user = message.from_user
    
    with log_context(user_id=user.id):
        logger.info("My stickers command received")
        
        try:
            # Get recent stickers
            stickers = await db_manager.get_user_stickers(user.id, limit=5)
            
            if not stickers:
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="🎨 Создать первый стикер",
                            callback_data=NavigationCallback(action="create_sticker").pack()
                        )
                    ]
                ])
                
                await message.answer(
                    "🎨 У вас пока нет созданных стикеров.\n"
                    "Самое время создать первый!",
                    reply_markup=keyboard
                )
                return
            
            # Send stickers as album
            media_group = []
            for i, sticker in enumerate(stickers):
                file_path = Path(sticker.file_path)
                if file_path.exists():
                    caption = None
                    if i == 0:  # First image gets the caption
                        caption = f"🎨 Ваши последние {len(stickers)} стикеров:"
                    
                    media_group.append(
                        InputMediaPhoto(
                            media=FSInputFile(str(file_path)),
                            caption=caption
                        )
                    )
            
            if media_group:
                await message.answer_media_group(media_group)
            
            # Send details and navigation
            details_text = "📝 *Детали стикеров:*\n\n"
            for i, sticker in enumerate(stickers, 1):
                details_text += (
                    f"{i}. _{sticker.prompt}_\n"
                    f"   📅 {sticker.created_at.strftime('%d.%m.%Y %H:%M')}\n"
                )
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="◀️ Предыдущие",
                        callback_data=NavigationCallback(action="stickers_page", target="prev").pack()
                    ),
                    InlineKeyboardButton(
                        text="Следующие ▶️",
                        callback_data=NavigationCallback(action="stickers_page", target="next").pack()
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="🔍 Поиск по стикерам",
                        callback_data=NavigationCallback(action="search_stickers").pack()
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="📥 Скачать все",
                        callback_data=NavigationCallback(action="download_all").pack()
                    )
                ]
            ])
            
            await message.answer(details_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Error getting stickers: {e}")
            await message.answer("❌ Не удалось получить ваши стикеры. Попробуйте позже.")


# === Message Handlers ===

@rate_limit_check
@check_user_access
async def voice_message_handler(
    message: Message,
    state: FSMContext,
    db_manager: DatabaseManager,
    generation_queue: GenerationQueue
):
    """Handle voice messages with transcription."""
    user = message.from_user
    
    with log_context(user_id=user.id, username=user.username):
        logger.info("Voice message received")
        
        if not message.voice:
            logger.warning("Message without voice data")
            await message.reply("❌ Не удалось получить голосовое сообщение.")
            return
        
        # Check file size
        file_size_mb = message.voice.file_size / (1024 * 1024)
        if file_size_mb > config.security.max_audio_size_mb:
            await message.reply(
                f"❌ Файл слишком большой ({file_size_mb:.1f} МБ).\n"
                f"Максимальный размер: {config.security.max_audio_size_mb} МБ"
            )
            return
        
        # Update stats
        await db_manager.update_user_request_stats(user.id, 'voice', user.username)
        
        # Show processing status
        status_msg = await message.answer("🎤 Распознаю голосовое сообщение...")
        
        # Download audio file
        audio_filename = f"voice_{user.id}_{uuid.uuid4().hex[:8]}.ogg"
        audio_file_path = config.paths.storage_dir / audio_filename
        
        try:
            # Download file
            file = await message.bot.get_file(message.voice.file_id)
            await message.bot.download_file(file.file_path, audio_file_path)
            logger.info(f"Audio file downloaded: {audio_file_path}")
            
            # Transcribe audio
            start_time = time.time()
            transcription_result = await transcribe_audio(str(audio_file_path))
            
            if not transcription_result or not transcription_result.get('text'):
                await status_msg.edit_text(
                    "❌ Не удалось распознать речь.\n"
                    "Попробуйте записать более четко или используйте текст."
                )
                await db_manager.log_error(
                    user.id, 'transcription',
                    'Failed to transcribe audio'
                )
                return
            
            prompt = transcription_result['text']
            language = transcription_result.get('language', 'unknown')
            process_time = transcription_result.get('process_time', 0)
            
            logger.info(
                f"Transcribed: '{prompt}' "
                f"[language: {language}, time: {process_time:.2f}s]"
            )
            
            # Update status
            await status_msg.edit_text(
                f"📝 Распознанный текст: *{prompt}*\n"
                f"🌐 Язык: {language}"
            )
            
            # Save prompt to state
            await state.update_data(
                prompt=prompt,
                request_type='voice',
                language=language
            )
            await state.set_state(StickerGeneration.waiting_for_style)
            
            # Show style selection
            await show_style_selection(message, state, db_manager)
            
        except TranscriptionError as e:
            logger.error(f"Transcription error: {e}")
            await status_msg.edit_text("❌ Ошибка распознавания речи.")
            await db_manager.log_error(user.id, 'transcription', str(e))
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            await status_msg.edit_text("❌ Произошла ошибка. Попробуйте позже.")
            await db_manager.log_error(user.id, 'voice_processing', str(e))
            
        finally:
            # Delete temporary file
            if audio_file_path.exists():
                try:
                    audio_file_path.unlink()
                    logger.debug(f"Temporary file deleted: {audio_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")


@rate_limit_check
@check_user_access
async def text_message_handler(
    message: Message,
    state: FSMContext,
    db_manager: DatabaseManager
):
    """Handle text messages."""
    user = message.from_user
    text = message.text.strip()
    
    with log_context(user_id=user.id, username=user.username):
        # Check if it's a command
        if text.startswith('/'):
            return
        
        # Check current state
        current_state = await state.get_state()
        
        # Handle keyboard buttons
        if text == "🎤 Голосовой стикер":
            await message.answer(
                "🎤 Отправьте голосовое сообщение с описанием стикера.\n"
                "Говорите четко и опишите, что должно быть на стикере."
            )
            await state.set_state(StickerGeneration.waiting_for_prompt)
            return
            
        elif text == "✍️ Текстовый стикер":
            await message.answer(
                "✍️ Напишите описание для вашего стикера.\n"
                "Например: «Милый котик с радугой» или «Космонавт на Луне»"
            )
            await state.set_state(StickerGeneration.waiting_for_prompt)
            return
            
        elif text == "🎨 Мои стикеры":
            await mystickers_command_handler(message, db_manager)
            return
            
        elif text == "📊 Статистика":
            await stats_command_handler(message, db_manager)
            return
            
        elif text == "⚙️ Настройки":
            await show_settings_menu(message, state, db_manager)
            return
            
        elif text == "📖 Помощь":
            await help_command_handler(message, db_manager)
            return
        
        # Handle text prompt
        if current_state in [None, StickerGeneration.waiting_for_prompt]:
            # Validate prompt
            if not text:
                await message.reply("❌ Пожалуйста, введите описание для стикера.")
                return
            
            if len(text) > 500:
                await message.reply(
                    "❌ Описание слишком длинное.\n"
                    "Максимум 500 символов."
                )
                return
            
            if len(text) < 3:
                await message.reply(
                    "❌ Описание слишком короткое.\n"
                    "Минимум 3 символа."
                )
                return
            
            logger.info(f"Text prompt received: '{text}'")
            
            # Update stats
            await db_manager.update_user_request_stats(user.id, 'text', user.username)
            
            # Save prompt to state
            await state.update_data(prompt=text, request_type='text')
            await state.set_state(StickerGeneration.waiting_for_style)
            
            # Show style selection
            await show_style_selection(message, state, db_manager)
            
        elif current_state == StickerGeneration.waiting_for_custom_style:
            # Handle custom style description
            await state.update_data(custom_style=text)
            await state.set_state(StickerGeneration.waiting_for_background)
            await show_background_selection(message, state, db_manager)
            
        elif current_state == StickerGeneration.editing_prompt:
            # Handle prompt editing
            await state.update_data(prompt=text)
            await state.set_state(StickerGeneration.waiting_for_style)
            await show_style_selection(message, state, db_manager)
            
        else:
            # Default response
            await message.answer(
                "Не понимаю команду. Используйте кнопки меню или /help для справки.",
                reply_markup=create_main_keyboard()
            )


# === Style and Background Selection ===

async def show_style_selection(
    message: Message,
    state: FSMContext,
    db_manager: DatabaseManager
):
    """Show style selection interface."""
    # Get user preferences
    user_prefs = await db_manager.get_user_preferences(message.from_user.id)
    default_style = user_prefs.get('default_style') if user_prefs else None
    
    # Get state data
    state_data = await state.get_data()
    prompt = state_data.get('prompt', '')
    
    await message.answer(
        f"🎨 Выберите стиль для стикера:\n\n"
        f"📝 Ваш промпт: _{prompt}_",
        reply_markup=create_style_keyboard(default_style)
    )


async def show_background_selection(
    message: Message,
    state: FSMContext,
    db_manager: DatabaseManager
):
    """Show background selection interface."""
    # Get user preferences
    user_prefs = await db_manager.get_user_preferences(message.from_user.id)
    default_bg = user_prefs.get('default_background') if user_prefs else None
    
    await message.answer(
        "🖼 Выберите фон для стикера:",
        reply_markup=create_background_keyboard(default_bg)
    )


async def show_settings_menu(
    message: Message,
    state: FSMContext,
    db_manager: DatabaseManager
):
    """Show user settings menu."""
    user = message.from_user
    
    # Get user preferences
    prefs = await db_manager.get_user_preferences(user.id) or {}
    
    settings_text = (
        "⚙️ *Настройки:*\n\n"
        f"🌐 Язык: {prefs.get('language', 'Русский')}\n"
        f"🎨 Стиль по умолчанию: {prefs.get('default_style', 'Не выбран')}\n"
        f"🖼 Фон по умолчанию: {prefs.get('default_background', 'Не выбран')}\n"
        f"🔔 Уведомления: {'Включены' if prefs.get('enable_notifications', True) else 'Выключены'}"
    )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🌐 Изменить язык",
                callback_data=NavigationCallback(action="change_language").pack()
            )
        ],
        [
            InlineKeyboardButton(
                text="🎨 Стиль по умолчанию",
                callback_data=NavigationCallback(action="default_style").pack()
            )
        ],
        [
            InlineKeyboardButton(
                text="🖼 Фон по умолчанию",
                callback_data=NavigationCallback(action="default_background").pack()
            )
        ],
        [
            InlineKeyboardButton(
                text="🔔 Уведомления",
                callback_data=NavigationCallback(action="toggle_notifications").pack()
            )
        ],
        [
            InlineKeyboardButton(
                text="❌ Закрыть",
                callback_data=NavigationCallback(action="close").pack()
            )
        ]
    ])
    
    await message.answer(settings_text, reply_markup=keyboard)
    await state.set_state(UserSettings.main_menu)


# === Callback Handlers ===

async def style_callback_handler(
    callback: CallbackQuery,
    callback_data: StyleCallback,
    state: FSMContext,
    db_manager: DatabaseManager,
    generation_queue: GenerationQueue
):
    """Handle style selection callbacks."""
    user = callback.from_user
    style_id = callback_data.style_id
    action = callback_data.action
    
    with log_context(user_id=user.id):
        logger.info(f"Style callback: {style_id}, action: {action}")
        
        if action == "select":
            if style_id == "custom":
                # Ask for custom style description
                await callback.message.edit_text(
                    "🎨 Опишите желаемый стиль своими словами.\n"
                    "Например: «в стиле Ван Гога» или «киберпанк с неоном»"
                )
                await state.set_state(StickerGeneration.waiting_for_custom_style)
            else:
                # Save selected style
                await state.update_data(style=style_id)
                await state.set_state(StickerGeneration.waiting_for_background)
                
                # Show background selection
                await callback.message.edit_text(
                    "🖼 Выберите фон для стикера:",
                    reply_markup=create_background_keyboard()
                )
        
        await callback.answer()


async def background_callback_handler(
    callback: CallbackQuery,
    callback_data: BackgroundCallback,
    state: FSMContext,
    db_manager: DatabaseManager,
    generation_queue: GenerationQueue
):
    """Handle background selection and start generation."""
    user = callback.from_user
    background_id = callback_data.background_id
    
    with log_context(user_id=user.id):
        logger.info(f"Background selected: {background_id}")
        
        # Get state data
        data = await state.get_data()
        prompt = data.get('prompt')
        style = data.get('style', 'default')
        custom_style = data.get('custom_style')
        
        if not prompt:
            await callback.message.edit_text(
                "❌ Ошибка: не найден текст для генерации."
            )
            await state.clear()
            await callback.answer()
            return
        
        # If custom style, append to prompt
        if style == "custom" and custom_style:
            prompt = f"{prompt}, {custom_style}"
        
        # Estimate generation time
        estimated_time = await estimate_generation_time(
            num_inference_steps=config.stable_diffusion.num_inference_steps
        )
        
        # Create generation request
        request_id = str(uuid.uuid4())
        generation_request = {
            'request_id': request_id,
            'user_id': user.id,
            'bot': callback.bot,
            'message': callback.message,
            'state': state,
            'prompt': prompt,
            'style': style,
            'background': background_id,
            'db_manager': db_manager,
            'estimated_time': estimated_time
        }
        
        # Add to queue
        position = await generation_queue.add_request(generation_request)
        
        # Update message
        if position > 1:
            await callback.message.edit_text(
                f"🔄 Ваш запрос добавлен в очередь.\n"
                f"Позиция: {position}\n"
                f"⏱ Примерное время ожидания: {position * estimated_time:.0f} сек"
            )
        else:
            await callback.message.edit_text(
                f"🎨 Начинаю генерацию стикера...\n"
                f"⏱ Примерное время: {estimated_time:.0f} сек"
            )
        
        await state.set_state(StickerGeneration.processing)
        await callback.answer()


async def navigation_callback_handler(
    callback: CallbackQuery,
    callback_data: NavigationCallback,
    state: FSMContext,
    db_manager: DatabaseManager
):
    """Handle navigation callbacks."""
    action = callback_data.action
    target = callback_data.target
    
    with log_context(user_id=callback.from_user.id):
        logger.info(f"Navigation: {action}, target: {target}")
        
        if action == "cancel":
            await state.clear()
            await callback.message.edit_text(
                "❌ Операция отменена.",
                reply_markup=None
            )
            await callback.message.answer(
                "Выберите действие:",
                reply_markup=create_main_keyboard()
            )
            
        elif action == "back":
            if target == "style":
                # Go back to style selection
                await state.set_state(StickerGeneration.waiting_for_style)
                await show_style_selection(callback.message, state, db_manager)
                
        elif action == "close":
            await callback.message.delete()
            
        # Add more navigation actions as needed
        
        await callback.answer()


async def sticker_callback_handler(
    callback: CallbackQuery,
    callback_data: StickerCallback,
    state: FSMContext,
    db_manager: DatabaseManager
):
    """Handle sticker action callbacks."""
    sticker_id = callback_data.sticker_id
    action = callback_data.action
    
    with log_context(user_id=callback.from_user.id):
        logger.info(f"Sticker action: {action} for {sticker_id}")
        
        if action == "regenerate":
            # Get sticker data and regenerate
            # Implementation depends on your needs
            await callback.answer("🔄 Функция регенерации в разработке", show_alert=True)
            
        elif action == "save":
            await callback.answer("💾 Стикер сохранен в вашей коллекции!")
            
        elif action == "share":
            # Create share link
            bot_username = (await callback.bot.get_me()).username
            share_url = f"https://t.me/{bot_username}?start=sticker_{sticker_id}"
            
            share_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📤 Поделиться",
                        url=f"https://t.me/share/url?url={share_url}&text=Посмотри какой классный стикер!"
                    )
                ]
            ])
            
            await callback.message.edit_reply_markup(reply_markup=share_keyboard)
            await callback.answer("📤 Используйте кнопку для отправки стикера")
            
        elif action == "delete":
            # Confirm deletion
            confirm_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Да, удалить",
                        callback_data=f"confirm_delete:{sticker_id}"
                    ),
                    InlineKeyboardButton(
                        text="❌ Отмена",
                        callback_data="cancel_delete"
                    )
                ]
            ])
            
            await callback.message.edit_reply_markup(reply_markup=confirm_keyboard)
            await callback.answer("⚠️ Подтвердите удаление")


# === Registration Function ===

def register_handlers(dp: Dispatcher):
    """Register all handlers with the dispatcher."""
    # Initialize components
    rate_limiter = RateLimiter()
    generation_queue = GenerationQueue(max_concurrent=config.security.max_concurrent_generations)
    
    # Store in dispatcher
    dp['rate_limiter'] = rate_limiter
    dp['generation_queue'] = generation_queue
    
    # Commands
    dp.message.register(start_command_handler, CommandStart())
    dp.message.register(help_command_handler, Command("help"))
    dp.message.register(stats_command_handler, Command("stats"))
    dp.message.register(mystickers_command_handler, Command("mystickers"))
    
    # Messages
    dp.message.register(voice_message_handler, F.content_type == ContentType.VOICE)
    dp.message.register(text_message_handler, F.content_type == ContentType.TEXT)
    
    # Callbacks
    dp.callback_query.register(
        style_callback_handler,
        StyleCallback.filter()
    )
    dp.callback_query.register(
        background_callback_handler,
        BackgroundCallback.filter()
    )
    dp.callback_query.register(
        navigation_callback_handler,
        NavigationCallback.filter()
    )
    dp.callback_query.register(
        sticker_callback_handler,
        StickerCallback.filter()
    )
    
    # Startup/shutdown
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)


async def on_startup(dispatcher: Dispatcher):
    """Execute on bot startup."""
    logger.info("Executing startup tasks...")
    
    # Start generation queue workers
    generation_queue = dispatcher['generation_queue']
    await generation_queue.start_workers()
    
    # Schedule periodic cleanup
    storage_dir = str(config.paths.storage_dir)
    asyncio.create_task(periodic_cleanup(storage_dir))
    
    logger.info("Startup tasks completed")


async def on_shutdown(dispatcher: Dispatcher):
    """Execute on bot shutdown."""
    logger.info("Executing shutdown tasks...")
    
    # Stop generation queue workers
    generation_queue = dispatcher['generation_queue']
    await generation_queue.stop_workers()
    
    logger.info("Shutdown tasks completed")


async def periodic_cleanup(storage_dir: str):
    """Periodic cleanup of old files."""
    while True:
        try:
            await asyncio.sleep(config.performance.cleanup_interval_hours * 3600)
            deleted = await cleanup_old_files(storage_dir, days=config.performance.cleanup_keep_days)
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old files")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")