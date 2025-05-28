from aiogram import Dispatcher, types, F  # <-- F импортируем из корневого модуля aiogram
from aiogram.filters import CommandStart, Command  # <-- F убираем отсюда!
from aiogram.types import ContentType, FSInputFile
from stt_service.service import transcribe_audio 
from image_generation_service.service import generate_sticker_image
from sticker_utils.utils import process_image_for_sticker, save_image_to_file
from db_manager.manager import DatabaseManager

import os
import uuid
import logging

logger = logging.getLogger(__name__)

def register_handlers(dp: Dispatcher):
    """Регистрирует все обработчики сообщений для бота."""
    dp.message.register(start_command_handler, CommandStart())
    dp.message.register(help_command_handler, Command("help"))
    dp.message.register(voice_message_handler, F.content_type == ContentType.VOICE)
    dp.message.register(text_message_handler, F.content_type == ContentType.TEXT)


async def start_command_handler(message: types.Message):
    user_name = message.from_user.first_name if message.from_user else "Пользователь"
    await message.answer(
        f"Привет, {user_name}! Я Голостикеры 👋\n\n"
        "Я могу генерировать стикеры по твоему голосовому запросу. "
        "Просто отправь мне голосовое сообщение с описанием того, какой стикер ты хочешь.\n\n"
        "Например: _'Грустный кот в шляпе'_ или _'Радостный единорог на радуге'_.\n\n"
        "Для подробной справки используй команду /help."
    )
    logger.info(f"Команда /start получена от пользователя {message.from_user.id}")

async def help_command_handler(message: types.Message):
    await message.answer(
        "👋 **Как пользоваться Голостикерами:**\n\n"
        "1. Запиши голосовое сообщение, описывающее стикер, который ты хочешь. Будь максимально конкретным!\n"
        "2. Отправь мне это голосовое сообщение.\n"
        "3. Я попытаюсь распознать твою речь и сгенерировать стикер.\n\n"
        "Если я не смогу распознать речь или сгенерировать стикер по голосовому сообщению, "
        "я попрошу тебя ввести описание текстом."
    )
    logger.info(f"Команда /help получена от пользователя {message.from_user.id}")

async def voice_message_handler(message: types.Message):
    dp_instance = message.bot.dispatcher
    logger = dp_instance["logger"]
    db_manager: DatabaseManager = dp_instance["db_manager"]
    storage_dir = dp_instance["storage_dir"]

    user_id = message.from_user.id
    chat_id = message.chat.id

    if not message.voice:
        logger.warning(f"Получено сообщение без голосовых данных от {user_id}")
        await message.reply("Что-то пошло не так с голосовым сообщением. Пожалуйста, попробуйте еще раз.")
        return

    logger.info(f"Получено голосовое сообщение от пользователя {user_id}")
    await message.bot.send_chat_action(chat_id, "typing")

    audio_file_id = message.voice.file_id
    audio_filename = f"{user_id}_{uuid.uuid4()}.ogg"
    audio_file_path = os.path.join(storage_dir, audio_filename)

    try:
        await message.bot.download(file=audio_file_id, destination=audio_file_path)
        logger.info(f"Аудиофайл {audio_file_path} скачан.")
    except Exception as e:
        logger.error(f"Ошибка при скачивании аудиофайла от {user_id}: {e}")
        await message.reply("Не удалось скачать ваше голосовое сообщение. Пожалуйста, попробуйте еще раз.")
        return

    await message.bot.send_message(chat_id, "Обрабатываю ваше голосовое сообщение...")
    try:
        prompt = await transcribe_audio(audio_file_path)
        if not prompt:
            await message.reply(
                "Не удалось распознать ваше голосовое сообщение. "
                "Пожалуйста, попробуйте отправить промт текстом."
            )
            logger.info(f"Не удалось распознать голосовое сообщение от {user_id}.")
            return

        logger.info(f"Распознано от {user_id}: {prompt}")
        await message.bot.send_message(chat_id, f"Распознал: \"*_{prompt}_*\". Начинаю генерировать стикер...", parse_mode="Markdown")
        await message.bot.send_chat_action(chat_id, "upload_photo")

    except Exception as e:
        logger.error(f"Ошибка распознавания речи для {user_id}: {e}")
        await message.reply(
            "Извините, произошла ошибка при распознавании речи. "
            "Пожалуйста, попробуйте позже или введите промт текстом."
        )
        return
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            logger.info(f"Временный аудиофайл {audio_file_path} удален.")

    generated_image_pil = None
    sticker_file_path = None
    try:
        generated_image_pil = await generate_sticker_image(prompt)
        if not generated_image_pil:
            await message.reply(
                "Не удалось сгенерировать стикер по вашему запросу. "
                "Попробуйте изменить описание или задать его текстом."
            )
            logger.info(f"Не удалось сгенерировать стикер для {user_id} по промту: {prompt}")
            return

        logger.info(f"Изображение сгенерировано для {user_id} по промту: {prompt}")
        processed_image_bytes = await process_image_for_sticker(generated_image_pil)

        sticker_filename = f"{user_id}_{uuid.uuid4()}.png"
        sticker_file_path = await save_image_to_file(processed_image_bytes, sticker_filename, storage_dir)
        logger.info(f"Стикер сохранен в {sticker_file_path}")

        await message.bot.send_photo(
            chat_id=chat_id,
            photo=FSInputFile(sticker_file_path),
            caption=f"Ваш стикер по запросу: *_{prompt}_*\n\n"
                    f"Вы можете добавить его в свой пак стикеров Telegram!"
        )
        logger.info(f"Стикер успешно отправлен пользователю {user_id}")

        await db_manager.add_sticker(user_id=user_id, prompt=prompt, file_path=sticker_file_path)
        logger.info(f"Информация о стикере для {user_id} добавлена в БД.")

    except Exception as e:
        logger.error(f"Произошла ошибка в процессе генерации/отправки стикера для {user_id} по промту '{prompt}': {e}")
        await message.reply(
            "Произошла ошибка при генерации или отправке стикера. "
            "Возможно, описание слишком сложное или содержит запрещенные слова. Пожалуйста, попробуйте еще раз."
        )
    finally:
        pass  # Можно добавить удаление файлов, если нужно

async def text_message_handler(message: types.Message):
    dp_instance = message.bot.dispatcher
    logger = dp_instance["logger"]
    db_manager: DatabaseManager = dp_instance["db_manager"]
    storage_dir = dp_instance["storage_dir"]

    user_id = message.from_user.id
    chat_id = message.chat.id
    prompt = message.text.strip()

    if not prompt:
        await message.reply("Пожалуйста, введите описание для стикера.")
        return

    logger.info(f"Получен текстовый промт от пользователя {user_id}: {prompt}")
    await message.bot.send_chat_action(chat_id, "upload_photo")

    generated_image_pil = None
    sticker_file_path = None
    try:
        generated_image_pil = await generate_sticker_image(prompt)
        if not generated_image_pil:
            await message.reply(
                "Не удалось сгенерировать стикер по вашему текстовому запросу. "
                "Попробуйте изменить описание."
            )
            logger.info(f"Не удалось сгенерировать стикер для {user_id} по текстовому промту: {prompt}")
            return

        logger.info(f"Изображение сгенерировано для {user_id} по текстовому промту: {prompt}")
        processed_image_bytes = await process_image_for_sticker(generated_image_pil)

        sticker_filename = f"{user_id}_{uuid.uuid4()}.png"
        sticker_file_path = await save_image_to_file(processed_image_bytes, sticker_filename, storage_dir)
        logger.info(f"Стикер сохранен в {sticker_file_path}")

        await message.bot.send_photo(
            chat_id=chat_id,
            photo=FSInputFile(sticker_file_path),
            caption=f"Ваш стикер по запросу: *_{prompt}_*\n\n"
                    f"Вы можете добавить его в свой пак стикеров Telegram!"
        )
        logger.info(f"Стикер успешно отправлен пользователю {user_id} (текстовый промт).")

        await db_manager.add_sticker(user_id=user_id, prompt=prompt, file_path=sticker_file_path)
        logger.info(f"Информация о стикере для {user_id} добавлена в БД (текстовый промт).")

    except Exception as e:
        logger.error(f"Произошла ошибка в процессе генерации/отправки стикера для {user_id} по текстовому промту '{prompt}': {e}")
        await message.reply(
            "Произошла ошибка при генерации или отправке стикера. "
            "Возможно, описание слишком сложное или содержит запрещенные слова. Пожалуйста, попробуйте еще раз."
        )
    finally:
        pass