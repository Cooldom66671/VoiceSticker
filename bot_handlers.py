"""
Обработчики сообщений для Telegram-бота "Голостикеры".
Управляет взаимодействием с пользователями.
"""
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from aiogram import Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import ContentType, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramBadRequest

from stt_service.service import transcribe_audio, TranscriptionError
from image_generation_service.service import (
    generate_sticker_image, GenerationError, 
    get_available_styles, estimate_generation_time
)
from sticker_utils.utils import (
    process_image_for_sticker, save_image_to_file,
    BackgroundStyle, cleanup_old_files
)
from db_manager.manager import DatabaseManager, DatabaseError
from logger import get_logger, LogContext
from config import MESSAGES, MAX_AUDIO_SIZE_MB

logger = get_logger(__name__)


class StickerGeneration(StatesGroup):
    """Состояния для генерации стикеров."""
    waiting_for_style = State()
    waiting_for_background = State()


def register_handlers(dp: Dispatcher):
    """Регистрирует все обработчики сообщений для бота."""
    # Команды
    dp.message.register(start_command_handler, CommandStart())
    dp.message.register(help_command_handler, Command("help"))
    dp.message.register(stats_command_handler, Command("stats"))
    dp.message.register(mystickers_command_handler, Command("mystickers"))
    
    # Сообщения
    dp.message.register(voice_message_handler, F.content_type == ContentType.VOICE)
    dp.message.register(text_message_handler, F.content_type == ContentType.TEXT)
    
    # Callback queries для inline кнопок
    dp.callback_query.register(style_callback_handler, F.data.startswith("style:"))
    dp.callback_query.register(background_callback_handler, F.data.startswith("bg:"))
    
    # Периодические задачи
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)


async def on_startup(dispatcher: Dispatcher):
    """Выполняется при запуске бота."""
    logger.info("Выполняю задачи при запуске...")
    
    # Очистка старых файлов
    storage_dir = dispatcher["storage_dir"]
    deleted = await cleanup_old_files(storage_dir, days=7)
    if deleted > 0:
        logger.info(f"Очищено {deleted} старых файлов")


async def on_shutdown(dispatcher: Dispatcher):
    """Выполняется при остановке бота."""
    logger.info("Выполняю задачи при остановке...")


async def start_command_handler(message: types.Message, state: FSMContext):
    """Обработчик команды /start."""
    user = message.from_user
    user_name = user.first_name or "Пользователь"
    
    # Получаем зависимости
    dp_instance = message.bot.dispatcher
    db_manager: DatabaseManager = dp_instance["db_manager"]
    
    # Логируем с контекстом пользователя
    with LogContext(logger, user_id=user.id, username=user.username):
        logger.info("Команда /start")
        
        # Обновляем статистику пользователя
        await db_manager.update_user_request_stats(
            user.id, 'text', user.username
        )
        
        # Создаем клавиатуру с полезными командами
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📖 Помощь", callback_data="cmd:help"),
                InlineKeyboardButton(text="📊 Статистика", callback_data="cmd:stats")
            ],
            [
                InlineKeyboardButton(text="🎨 Мои стикеры", callback_data="cmd:mystickers")
            ]
        ])
        
        # Отправляем приветствие
        await message.answer(
            MESSAGES["start"].format(name=user_name),
            reply_markup=keyboard
        )
        
        # Сбрасываем состояние
        await state.clear()


async def help_command_handler(message: types.Message):
    """Обработчик команды /help."""
    user = message.from_user
    
    with LogContext(logger, user_id=user.id):
        logger.info("Команда /help")
        
        # Список доступных стилей
        styles = await get_available_styles()
        styles_text = "\n".join([f"   • {s['name']}: {s['description']}" for s in styles])
        
        # Расширенная справка
        help_text = MESSAGES["help"] + f"\n\n🎨 *Доступные стили:*\n{styles_text}"
        
        await message.answer(help_text)


async def stats_command_handler(message: types.Message):
    """Обработчик команды /stats - показывает статистику пользователя."""
    user = message.from_user
    dp_instance = message.bot.dispatcher
    db_manager: DatabaseManager = dp_instance["db_manager"]
    
    with LogContext(logger, user_id=user.id):
        logger.info("Команда /stats")
        
        try:
            # Получаем статистику пользователя
            user_stats = await db_manager.get_user_stats(user.id)
            
            if not user_stats:
                await message.answer("📊 У вас пока нет статистики. Создайте свой первый стикер!")
                return
            
            # Форматируем статистику
            stats_text = (
                "📊 *Ваша статистика:*\n\n"
                f"🎨 Стикеров создано: {user_stats['total_stickers']}\n"
                f"🎤 Голосовых запросов: {user_stats['total_voice_requests']}\n"
                f"✍️ Текстовых запросов: {user_stats['total_text_requests']}\n"
                f"📅 Первое использование: {user_stats['first_use'][:10]}\n"
                f"🕐 Последнее использование: {user_stats['last_use'][:10]}"
            )
            
            await message.answer(stats_text)
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            await message.answer("❌ Не удалось получить статистику. Попробуйте позже.")


async def mystickers_command_handler(message: types.Message):
    """Обработчик команды /mystickers - показывает последние стикеры пользователя."""
    user = message.from_user
    dp_instance = message.bot.dispatcher
    db_manager: DatabaseManager = dp_instance["db_manager"]
    
    with LogContext(logger, user_id=user.id):
        logger.info("Команда /mystickers")
        
        try:
            # Получаем последние стикеры
            stickers = await db_manager.get_user_stickers(user.id, limit=5)
            
            if not stickers:
                await message.answer("🎨 У вас пока нет созданных стикеров. Самое время создать первый!")
                return
            
            await message.answer(f"🎨 Ваши последние {len(stickers)} стикеров:")
            
            # Отправляем стикеры
            for sticker in stickers:
                file_path = Path(sticker['file_path'])
                if file_path.exists():
                    caption = f"📝 _{sticker['prompt']}_\n⏱ {sticker['created_at'][:16]}"
                    
                    try:
                        await message.answer_photo(
                            photo=FSInputFile(str(file_path)),
                            caption=caption
                        )
                    except Exception as e:
                        logger.warning(f"Не удалось отправить стикер {file_path}: {e}")
                
            await message.answer("💡 Используйте /start для создания новых стикеров!")
            
        except Exception as e:
            logger.error(f"Ошибка получения стикеров: {e}")
            await message.answer("❌ Не удалось получить ваши стикеры. Попробуйте позже.")


async def voice_message_handler(message: types.Message, state: FSMContext):
    """Обработчик голосовых сообщений."""
    user = message.from_user
    dp_instance = message.bot.dispatcher
    db_manager: DatabaseManager = dp_instance["db_manager"]
    storage_dir = dp_instance["storage_dir"]
    
    with LogContext(logger, user_id=user.id, username=user.username):
        logger.info("Получено голосовое сообщение")
        
        # Проверяем наличие голосовых данных
        if not message.voice:
            logger.warning("Сообщение без голосовых данных")
            await message.reply("❌ Не удалось получить голосовое сообщение. Попробуйте еще раз.")
            return
        
        # Проверяем размер файла
        file_size_mb = message.voice.file_size / (1024 * 1024)
        if file_size_mb > MAX_AUDIO_SIZE_MB:
            await message.reply(MESSAGES["error_file_too_large"].format(MAX_AUDIO_SIZE_MB))
            return
        
        # Обновляем статистику
        await db_manager.update_user_request_stats(user.id, 'voice', user.username)
        
        # Показываем статус обработки
        status_msg = await message.answer(MESSAGES["processing_voice"])
        await message.bot.send_chat_action(message.chat.id, "typing")
        
        # Скачиваем аудиофайл
        audio_filename = f"voice_{user.id}_{uuid.uuid4().hex[:8]}.ogg"
        audio_file_path = os.path.join(storage_dir, audio_filename)
        
        try:
            # Скачиваем файл
            file = await message.bot.get_file(message.voice.file_id)
            await message.bot.download_file(file.file_path, audio_file_path)
            logger.info(f"Аудиофайл скачан: {audio_file_path}")
            
            # Распознаем речь
            start_time = time.time()
            transcription_result = await transcribe_audio(audio_file_path)
            
            if not transcription_result or not transcription_result.get('text'):
                await status_msg.edit_text(MESSAGES["error_transcription"])
                await db_manager.log_error(
                    user.id, 'transcription', 
                    'Не удалось распознать речь'
                )
                return
            
            prompt = transcription_result['text']
            language = transcription_result.get('language', 'unknown')
            process_time = transcription_result.get('process_time', 0)
            
            logger.info(
                f"Распознано: '{prompt}' "
                f"[язык: {language}, время: {process_time:.2f}с]"
            )
            
            # Обновляем статус
            await status_msg.edit_text(
                MESSAGES["transcription_result"].format(prompt)
            )
            
            # Сохраняем промпт в состоянии
            await state.update_data(prompt=prompt, request_type='voice')
            
            # Показываем выбор стиля
            await show_style_selection(message, state)
            
        except TranscriptionError as e:
            logger.error(f"Ошибка транскрибации: {e}")
            await status_msg.edit_text(MESSAGES["error_transcription"])
            await db_manager.log_error(user.id, 'transcription', str(e))
            
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
            await status_msg.edit_text(MESSAGES["error_processing"])
            await db_manager.log_error(user.id, 'voice_processing', str(e))
            
        finally:
            # Удаляем временный файл
            if os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                    logger.debug(f"Временный файл удален: {audio_file_path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл: {e}")


async def text_message_handler(message: types.Message, state: FSMContext):
    """Обработчик текстовых сообщений."""
    user = message.from_user
    prompt = message.text.strip()
    
    with LogContext(logger, user_id=user.id, username=user.username):
        # Проверяем, не команда ли это
        if prompt.startswith('/'):
            return
        
        # Проверяем длину промпта
        if not prompt:
            await message.reply("❌ Пожалуйста, введите описание для стикера.")
            return
        
        if len(prompt) > 500:
            await message.reply("❌ Описание слишком длинное. Максимум 500 символов.")
            return
        
        logger.info(f"Получен текстовый промпт: '{prompt}'")
        
        # Получаем зависимости
        dp_instance = message.bot.dispatcher
        db_manager: DatabaseManager = dp_instance["db_manager"]
        
        # Обновляем статистику
        await db_manager.update_user_request_stats(user.id, 'text', user.username)
        
        # Сохраняем промпт в состоянии
        await state.update_data(prompt=prompt, request_type='text')
        
        # Показываем выбор стиля
        await show_style_selection(message, state)


async def show_style_selection(message: types.Message, state: FSMContext):
    """Показывает выбор стиля для генерации."""
    # Получаем доступные стили
    styles = await get_available_styles()
    
    # Создаем клавиатуру
    keyboard = InlineKeyboardMarkup(inline_keyboard=[])
    
    # Добавляем кнопки стилей (по 2 в ряд)
    for i in range(0, len(styles), 2):
        row = []
        for j in range(2):
            if i + j < len(styles):
                style = styles[i + j]
                row.append(InlineKeyboardButton(
                    text=style['name'],
                    callback_data=f"style:{style['id']}"
                ))
        keyboard.inline_keyboard.append(row)
    
    # Добавляем кнопку отмены
    keyboard.inline_keyboard.append([
        InlineKeyboardButton(text="❌ Отменить", callback_data="cancel")
    ])
    
    await message.answer(
        "🎨 Выберите стиль для вашего стикера:",
        reply_markup=keyboard
    )
    
    # Устанавливаем состояние
    await state.set_state(StickerGeneration.waiting_for_style)


async def style_callback_handler(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик выбора стиля."""
    user = callback.from_user
    style_id = callback.data.split(":")[1]
    
    with LogContext(logger, user_id=user.id):
        logger.info(f"Выбран стиль: {style_id}")
        
        # Сохраняем выбранный стиль
        await state.update_data(style=style_id)
        
        # Показываем выбор фона
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🏻 Прозрачный", callback_data="bg:transparent"),
                InlineKeyboardButton(text="⬜ Белый", callback_data="bg:white")
            ],
            [
                InlineKeyboardButton(text="🌈 Градиент", callback_data="bg:gradient"),
                InlineKeyboardButton(text="⭕ Круглый", callback_data="bg:circle")
            ],
            [
                InlineKeyboardButton(text="❌ Отменить", callback_data="cancel")
            ]
        ])
        
        await callback.message.edit_text(
            "🖼 Выберите фон для стикера:",
            reply_markup=keyboard
        )
        
        await state.set_state(StickerGeneration.waiting_for_background)
        await callback.answer()


async def background_callback_handler(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик выбора фона и генерация стикера."""
    user = callback.from_user
    background = callback.data.split(":")[1]
    
    # Получаем зависимости
    dp_instance = callback.bot.dispatcher
    db_manager: DatabaseManager = dp_instance["db_manager"]
    storage_dir = dp_instance["storage_dir"]
    
    with LogContext(logger, user_id=user.id):
        logger.info(f"Выбран фон: {background}")
        
        # Получаем данные из состояния
        data = await state.get_data()
        prompt = data.get('prompt')
        style = data.get('style', 'default')
        
        if not prompt:
            await callback.message.edit_text("❌ Ошибка: не найден текст для генерации.")
            await state.clear()
            return
        
        # Оцениваем время генерации
        estimated_time = await estimate_generation_time(num_inference_steps=30)
        
        # Обновляем сообщение
        await callback.message.edit_text(
            f"🎨 Генерирую стикер...\n"
            f"📝 Промпт: _{prompt}_\n"
            f"🎨 Стиль: {style}\n"
            f"🖼 Фон: {background}\n"
            f"⏱ Примерное время: {estimated_time:.0f} сек"
        )
        
        await callback.bot.send_chat_action(callback.message.chat.id, "upload_photo")
        
        # Генерируем изображение
        start_time = time.time()
        
        try:
            # Генерируем изображение
            generated_image = await generate_sticker_image(
                prompt=prompt,
                style=style,
                enhance_prompt=True
            )
            
            if not generated_image:
                await callback.message.edit_text(MESSAGES["error_generation"])
                await db_manager.log_error(user.id, 'generation', f'Не удалось сгенерировать: {prompt}')
                await state.clear()
                return
            
            generation_time = time.time() - start_time
            logger.info(f"Изображение сгенерировано за {generation_time:.2f}с")
            
            # Обрабатываем изображение для стикера
            background_style = BackgroundStyle(background)
            processed_image = await process_image_for_sticker(
                generated_image,
                background_style=background_style,
                add_shadow=background in ['transparent', 'circle'],
                add_outline=style in ['cartoon', 'anime'],
                remove_white_bg=background == 'transparent'
            )
            
            if not processed_image:
                await callback.message.edit_text(MESSAGES["error_processing"])
                await state.clear()
                return
            
            # Сохраняем файл
            sticker_filename = f"sticker_{user.id}_{uuid.uuid4().hex[:8]}.png"
            sticker_path = await save_image_to_file(
                processed_image,
                sticker_filename,
                storage_dir
            )
            
            if not sticker_path:
                await callback.message.edit_text(MESSAGES["error_processing"])
                await state.clear()
                return
            
            # Определяем размер файла
            file_size = os.path.getsize(sticker_path) / 1024  # в КБ
            
            # Отправляем стикер
            await callback.bot.send_photo(
                chat_id=callback.message.chat.id,
                photo=FSInputFile(sticker_path),
                caption=(
                    f"✅ {MESSAGES['sticker_ready']}\n\n"
                    f"📝 Промпт: _{prompt}_\n"
                    f"🎨 Стиль: {style}\n"
                    f"🖼 Фон: {background}\n"
                    f"📦 Размер: {file_size:.1f} КБ\n"
                    f"⏱ Время генерации: {generation_time:.1f} сек"
                )
            )
            
            # Сохраняем в БД
            await db_manager.add_sticker(
                user_id=user.id,
                prompt=prompt,
                file_path=sticker_path,
                username=user.username,
                file_size=int(file_size * 1024),
                generation_time=generation_time,
                model_version=f"SD1.5-{style}"
            )
            
            # Удаляем сообщение с прогрессом
            try:
                await callback.message.delete()
            except TelegramBadRequest:
                pass
            
            logger.info(f"Стикер успешно создан и отправлен")
            
        except GenerationError as e:
            logger.error(f"Ошибка генерации: {e}")
            await callback.message.edit_text(MESSAGES["error_generation"])
            await db_manager.log_error(user.id, 'generation', str(e), prompt)
            
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
            await callback.message.edit_text(MESSAGES["error_processing"])
            await db_manager.log_error(user.id, 'processing', str(e), prompt)
            
        finally:
            # Очищаем состояние
            await state.clear()
            await callback.answer()