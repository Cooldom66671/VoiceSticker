import asyncio
import logging
import os
import whisper # Импортируем библиотеку whisper
from pydub import AudioSegment # pydub для обработки аудио

logger = logging.getLogger(__name__)

# Глобальная переменная для модели Whisper. Загружаем модель один раз.
_whisper_model = None

async def load_whisper_model():
    """Асинхронно загружает модель Whisper."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Загрузка модели OpenAI Whisper ('base' model)...")
        # Выбирай модель: "tiny", "base", "small", "medium", "large"
        # "base" - хороший баланс между размером и качеством, достаточно производителен на CPU, на GPU - очень быстро.
        # Для M1/M2/M3 чипов Whisper также может использовать MPS (Metal Performance Shaders)
        # load_model - это блокирующая операция, выполняем ее в отдельном потоке
        _whisper_model = await asyncio.to_thread(whisper.load_model, "base")
        logger.info("Модель OpenAI Whisper загружена.")
    return _whisper_model


async def transcribe_audio(audio_file_path: str) -> str | None:
    """
    Распознает речь из аудиофайла с использованием OpenAI Whisper.
    Поддерживает OGG-файлы от Telegram, конвертируя их в формат, понятный Whisper (если требуется),
    или передавая напрямую.
    """
    logger.info(f"Начинаю распознавание аудиофайла с Whisper: {audio_file_path}")

    model = await load_whisper_model() # Убеждаемся, что модель загружена

    try:
        # Whisper может напрямую работать с OGG файлами, если ffmpeg установлен и доступен.
        # Если возникают проблемы, можно явно конвертировать в WAV/MP3 через pydub.
        # Для начала попробуем напрямую передать OGG, т.к. Whisper хорошо справляется.

        # fp16=False для работы на CPU. Если есть GPU с поддержкой FP16, можно установить в True.
        # language=None для автоматического определения языка (ru/en).
        result = await asyncio.to_thread(model.transcribe, audio_file_path, language=None, fp16=False)
        
        transcript = result["text"].strip() # Убираем лишние пробелы в начале/конце
        
        if transcript:
            logger.info(f"Whisper распознал: {transcript}")
            return transcript
        else:
            logger.info("Whisper не смог распознать речь или распознал пустой текст.")
            return None
    except Exception as e:
        logger.error(f"Ошибка при распознавании с OpenAI Whisper для {audio_file_path}: {e}")
        return None
    finally:
        # Временный аудиофайл (.ogg) будет удален в bot_handlers,
        # так что здесь нет необходимости его удалять.
        pass