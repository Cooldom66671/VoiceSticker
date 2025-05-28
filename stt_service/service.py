"""
Сервис распознавания речи с использованием OpenAI Whisper.
Поддерживает различные аудиоформаты и языки.
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache
import tempfile
import shutil

import whisper
import torch
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from logger import get_logger
from config import (
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_TIMEOUT,
    MAX_AUDIO_SIZE_MB, MAX_AUDIO_DURATION_SEC,
    ALLOWED_AUDIO_FORMATS, get_device_config
)

logger = get_logger(__name__)

# Глобальные переменные для модели
_whisper_model: Optional[whisper.Whisper] = None
_model_lock = asyncio.Lock()


class TranscriptionError(Exception):
    """Исключение для ошибок транскрибации."""
    pass


class AudioProcessor:
    """Класс для предобработки аудиофайлов."""
    
    @staticmethod
    async def validate_audio(file_path: str) -> Dict[str, Any]:
        """
        Валидирует аудиофайл.
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Словарь с информацией о файле
            
        Raises:
            TranscriptionError: Если файл не прошел валидацию
        """
        path = Path(file_path)
        
        # Проверяем существование файла
        if not path.exists():
            raise TranscriptionError(f"Файл не найден: {file_path}")
        
        # Проверяем размер файла
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_AUDIO_SIZE_MB:
            raise TranscriptionError(
                f"Файл слишком большой: {file_size_mb:.1f} МБ "
                f"(максимум: {MAX_AUDIO_SIZE_MB} МБ)"
            )
        
        # Проверяем формат файла
        file_ext = path.suffix.lower()
        if file_ext not in ALLOWED_AUDIO_FORMATS:
            raise TranscriptionError(
                f"Неподдерживаемый формат: {file_ext}. "
                f"Поддерживаются: {', '.join(ALLOWED_AUDIO_FORMATS)}"
            )
        
        # Получаем информацию об аудио
        try:
            audio = await asyncio.to_thread(AudioSegment.from_file, file_path)
            duration_sec = len(audio) / 1000.0
            
            if duration_sec > MAX_AUDIO_DURATION_SEC:
                raise TranscriptionError(
                    f"Аудио слишком длинное: {duration_sec:.1f} сек "
                    f"(максимум: {MAX_AUDIO_DURATION_SEC} сек)"
                )
            
            return {
                'path': str(path),
                'size_mb': file_size_mb,
                'duration_sec': duration_sec,
                'channels': audio.channels,
                'frame_rate': audio.frame_rate,
                'sample_width': audio.sample_width,
                'format': file_ext
            }
            
        except CouldntDecodeError as e:
            raise TranscriptionError(f"Не удалось декодировать аудио: {e}")
        except Exception as e:
            raise TranscriptionError(f"Ошибка при анализе аудио: {e}")
    
    @staticmethod
    async def prepare_audio_for_whisper(
        file_path: str,
        target_sample_rate: int = 16000
    ) -> str:
        """
        Подготавливает аудио для Whisper (конвертация в оптимальный формат).
        
        Args:
            file_path: Путь к исходному файлу
            target_sample_rate: Целевая частота дискретизации
            
        Returns:
            Путь к подготовленному файлу
        """
        try:
            # Загружаем аудио
            audio = await asyncio.to_thread(AudioSegment.from_file, file_path)
            
            # Конвертируем в моно если стерео
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.debug("Конвертировано в моно")
            
            # Изменяем частоту дискретизации если нужно
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
                logger.debug(f"Частота дискретизации изменена на {target_sample_rate} Hz")
            
            # Нормализуем громкость
            audio = audio.normalize()
            
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                dir=Path(file_path).parent
            ) as tmp_file:
                tmp_path = tmp_file.name
                
            # Экспортируем в WAV
            await asyncio.to_thread(
                audio.export,
                tmp_path,
                format='wav',
                parameters=['-q:a', '0']  # Максимальное качество
            )
            
            logger.debug(f"Аудио подготовлено: {tmp_path}")
            return tmp_path
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке аудио: {e}")
            raise TranscriptionError(f"Не удалось подготовить аудио: {e}")


async def load_whisper_model() -> whisper.Whisper:
    """
    Загружает модель Whisper (однократно).
    
    Returns:
        Загруженная модель Whisper
    """
    global _whisper_model
    
    async with _model_lock:
        if _whisper_model is None:
            start_time = time.time()
            logger.info(f"Загрузка модели Whisper '{WHISPER_MODEL}'...")
            
            try:
                # Определяем устройство
                device_config = get_device_config()
                device = device_config['whisper_device']
                
                # Загружаем модель в отдельном потоке
                _whisper_model = await asyncio.to_thread(
                    whisper.load_model,
                    WHISPER_MODEL,
                    device=device,
                    download_root=".cache/whisper"
                )
                
                load_time = time.time() - start_time
                logger.info(
                    f"Модель Whisper '{WHISPER_MODEL}' загружена за {load_time:.2f} сек "
                    f"на устройстве '{device}'"
                )
                
                # Прогреваем модель
                await _warmup_model(_whisper_model)
                
            except Exception as e:
                logger.error(f"Ошибка загрузки модели Whisper: {e}")
                raise TranscriptionError(f"Не удалось загрузить модель: {e}")
    
    return _whisper_model


async def _warmup_model(model: whisper.Whisper):
    """Прогревает модель для ускорения первого вызова."""
    try:
        logger.debug("Прогрев модели Whisper...")
        
        # Создаем короткий тестовый аудиофайл
        test_audio = AudioSegment.silent(duration=1000)  # 1 секунда тишины
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            test_audio.export(tmp.name, format='wav')
            
            # Выполняем тестовую транскрибацию
            await asyncio.to_thread(
                model.transcribe,
                tmp.name,
                language='en',
                fp16=False
            )
        
        logger.debug("Модель прогрета")
        
    except Exception as e:
        logger.warning(f"Не удалось прогреть модель: {e}")


async def transcribe_audio(
    audio_file_path: str,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    temperature: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Распознает речь из аудиофайла.
    
    Args:
        audio_file_path: Путь к аудиофайлу
        language: Код языка (None для автоопределения)
        initial_prompt: Начальный промт для улучшения распознавания
        temperature: Температура генерации (0.0 для детерминированного результата)
        
    Returns:
        Словарь с результатами транскрибации или None при ошибке
    """
    start_time = time.time()
    temp_file = None
    
    try:
        # Валидируем аудиофайл
        audio_info = await AudioProcessor.validate_audio(audio_file_path)
        logger.info(
            f"Транскрибация аудио: {audio_info['duration_sec']:.1f} сек, "
            f"{audio_info['size_mb']:.1f} МБ, {audio_info['format']}"
        )
        
        # Загружаем модель
        model = await load_whisper_model()
        
        # Подготавливаем аудио если нужно
        if audio_info['format'] == '.ogg' or audio_info['frame_rate'] != 16000:
            logger.debug("Конвертация аудио для Whisper...")
            temp_file = await AudioProcessor.prepare_audio_for_whisper(audio_file_path)
            process_file = temp_file
        else:
            process_file = audio_file_path
        
        # Определяем параметры для устройства
        device_config = get_device_config()
        use_fp16 = (
            device_config['whisper_device'] in ['cuda', 'mps'] and
            torch.cuda.is_available() if device_config['whisper_device'] == 'cuda' else True
        )
        
        # Выполняем транскрибацию с таймаутом
        logger.debug(f"Начинаю транскрибацию на устройстве '{device_config['whisper_device']}'...")
        
        transcribe_task = asyncio.create_task(
            asyncio.to_thread(
                model.transcribe,
                process_file,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
                fp16=use_fp16,
                verbose=False,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
        )
        
        result = await asyncio.wait_for(transcribe_task, timeout=WHISPER_TIMEOUT)
        
        # Обрабатываем результат
        transcript = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")
        
        if not transcript:
            logger.warning("Whisper не распознал речь или текст пустой")
            return None
        
        # Очищаем текст от артефактов
        transcript = _clean_transcript(transcript)
        
        process_time = time.time() - start_time
        logger.info(
            f"Транскрибация завершена за {process_time:.2f} сек. "
            f"Язык: {detected_language}, длина: {len(transcript)} символов"
        )
        
        return {
            'text': transcript,
            'language': detected_language,
            'duration': audio_info['duration_sec'],
            'process_time': process_time,
            'segments': result.get('segments', [])  # Детальная информация по сегментам
        }
        
    except asyncio.TimeoutError:
        logger.error(f"Превышен таймаут транскрибации ({WHISPER_TIMEOUT} сек)")
        return None
        
    except TranscriptionError as e:
        logger.error(f"Ошибка валидации: {e}")
        return None
        
    except Exception as e:
        logger.error(f"Неожиданная ошибка при транскрибации: {e}", exc_info=True)
        return None
        
    finally:
        # Удаляем временный файл
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.debug(f"Временный файл удален: {temp_file}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл: {e}")


def _clean_transcript(text: str) -> str:
    """
    Очищает транскрипт от типичных артефактов Whisper.
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    # Удаляем повторяющиеся пробелы
    text = ' '.join(text.split())
    
    # Удаляем типичные артефакты Whisper
    artifacts = [
        '[BLANK_AUDIO]',
        '(unintelligible)',
        '(inaudible)',
        '[Music]',
        '[Applause]',
        '...',
        '♪'
    ]
    
    for artifact in artifacts:
        text = text.replace(artifact, '')
    
    # Удаляем лишние пробелы после очистки
    text = ' '.join(text.split()).strip()
    
    return text


# Дополнительные утилиты
async def get_audio_languages(audio_file_path: str) -> Dict[str, float]:
    """
    Определяет вероятности языков в аудио.
    
    Args:
        audio_file_path: Путь к аудиофайлу
        
    Returns:
        Словарь {язык: вероятность}
    """
    try:
        model = await load_whisper_model()
        
        # Whisper может определить язык
        audio = whisper.load_audio(audio_file_path)
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        _, probs = await asyncio.to_thread(
            model.detect_language,
            mel
        )
        
        # Сортируем по вероятности
        languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return dict(languages[:5])  # Топ-5 языков
        
    except Exception as e:
        logger.error(f"Ошибка определения языка: {e}")
        return {}