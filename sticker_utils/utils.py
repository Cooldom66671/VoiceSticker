"""
Утилиты для обработки изображений и создания стикеров.
Оптимизировано для требований Telegram.
"""
import asyncio
import io
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import uuid

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops
import numpy as np

from logger import get_logger
from config import STICKER_MAX_SIZE, STICKER_FORMAT, STICKER_QUALITY

logger = get_logger(__name__)


class StickerError(Exception):
    """Исключение для ошибок обработки стикеров."""
    pass


class BackgroundStyle(Enum):
    """Стили фона для стикеров."""
    TRANSPARENT = "transparent"
    WHITE = "white"
    GRADIENT = "gradient"
    CIRCLE = "circle"
    ROUNDED = "rounded"


class ImageProcessor:
    """Класс для продвинутой обработки изображений."""
    
    @staticmethod
    def remove_background_simple(image: Image.Image, threshold: int = 240) -> Image.Image:
        """
        Простое удаление белого/светлого фона.
        
        Args:
            image: Исходное изображение
            threshold: Порог для определения белого цвета (0-255)
            
        Returns:
            Изображение с прозрачным фоном
        """
        # Конвертируем в RGBA если нужно
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Создаем массив numpy для быстрой обработки
        data = np.array(image)
        
        # Находим белые/светлые пиксели
        white_areas = (data[:, :, :3] > threshold).all(axis=2)
        
        # Делаем их прозрачными
        data[white_areas, 3] = 0
        
        return Image.fromarray(data, 'RGBA')
    
    @staticmethod
    def add_shadow(image: Image.Image, offset: Tuple[int, int] = (5, 5), 
                  blur_radius: int = 5, opacity: int = 128) -> Image.Image:
        """
        Добавляет тень к изображению.
        
        Args:
            image: Исходное изображение
            offset: Смещение тени (x, y)
            blur_radius: Радиус размытия тени
            opacity: Прозрачность тени (0-255)
            
        Returns:
            Изображение с тенью
        """
        # Создаем изображение большего размера для тени
        shadow_padding = max(abs(offset[0]), abs(offset[1])) + blur_radius * 2
        new_size = (
            image.width + shadow_padding * 2,
            image.height + shadow_padding * 2
        )
        
        # Создаем холст с прозрачным фоном
        result = Image.new('RGBA', new_size, (0, 0, 0, 0))
        
        # Создаем тень
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Рисуем черную форму объекта
        if image.mode == 'RGBA':
            # Используем альфа-канал как маску
            shadow.paste((0, 0, 0, opacity), mask=image.split()[3])
        else:
            shadow = Image.new('RGBA', image.size, (0, 0, 0, opacity))
        
        # Размываем тень
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Вставляем тень со смещением
        shadow_pos = (shadow_padding + offset[0], shadow_padding + offset[1])
        result.paste(shadow, shadow_pos)
        
        # Вставляем оригинальное изображение
        image_pos = (shadow_padding, shadow_padding)
        result.paste(image, image_pos, image if image.mode == 'RGBA' else None)
        
        return result
    
    @staticmethod
    def add_outline(image: Image.Image, outline_size: int = 3, 
                   outline_color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
        """
        Добавляет обводку к изображению.
        
        Args:
            image: Исходное изображение
            outline_size: Толщина обводки в пикселях
            outline_color: Цвет обводки (RGBA)
            
        Returns:
            Изображение с обводкой
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Создаем маску из альфа-канала
        mask = image.split()[3]
        
        # Расширяем маску для создания обводки
        outline = mask.filter(ImageFilter.MaxFilter(outline_size * 2 + 1))
        
        # Создаем новое изображение с обводкой
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
        # Рисуем обводку
        result.paste(outline_color, mask=outline)
        
        # Вставляем оригинальное изображение поверх
        result.paste(image, (0, 0), image)
        
        return result
    
    @staticmethod
    def create_circular_mask(size: Tuple[int, int]) -> Image.Image:
        """
        Создает круглую маску.
        
        Args:
            size: Размер маски (ширина, высота)
            
        Returns:
            Маска в виде изображения
        """
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + size, fill=255)
        return mask
    
    @staticmethod
    def create_rounded_mask(size: Tuple[int, int], radius: int) -> Image.Image:
        """
        Создает маску с закругленными углами.
        
        Args:
            size: Размер маски
            radius: Радиус закругления
            
        Returns:
            Маска в виде изображения
        """
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Рисуем прямоугольник с закругленными углами
        draw.rounded_rectangle(
            [(0, 0), size],
            radius=radius,
            fill=255
        )
        
        return mask


async def process_image_for_sticker(
    pil_image: Image.Image,
    background_style: BackgroundStyle = BackgroundStyle.TRANSPARENT,
    add_shadow: bool = False,
    add_outline: bool = False,
    remove_white_bg: bool = False,
    optimize_size: bool = True
) -> Optional[io.BytesIO]:
    """
    Обрабатывает изображение для использования в качестве стикера Telegram.
    
    Args:
        pil_image: Исходное изображение PIL
        background_style: Стиль фона
        add_shadow: Добавить ли тень
        add_outline: Добавить ли обводку
        remove_white_bg: Удалить ли белый фон
        optimize_size: Оптимизировать ли размер файла
        
    Returns:
        BytesIO с обработанным изображением или None при ошибке
    """
    logger.info(f"Обработка изображения для стикера [фон: {background_style.value}]")
    
    try:
        # Конвертируем в RGBA для работы с прозрачностью
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        # Удаляем белый фон если требуется
        if remove_white_bg:
            pil_image = await asyncio.to_thread(
                ImageProcessor.remove_background_simple,
                pil_image
            )
        
        # Добавляем эффекты
        if add_outline:
            pil_image = await asyncio.to_thread(
                ImageProcessor.add_outline,
                pil_image
            )
        
        if add_shadow:
            pil_image = await asyncio.to_thread(
                ImageProcessor.add_shadow,
                pil_image
            )
        
        # Вычисляем размеры с сохранением пропорций
        original_width, original_height = pil_image.size
        max_size = STICKER_MAX_SIZE
        
        # Вычисляем коэффициент масштабирования
        scale = min(max_size / original_width, max_size / original_height)
        
        # Если изображение меньше максимального размера, не увеличиваем
        if scale > 1.0:
            scale = 1.0
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Изменяем размер с высоким качеством
        resized_image = await asyncio.to_thread(
            pil_image.resize,
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        # Создаем финальное изображение с нужным фоном
        final_image = await _create_sticker_background(
            resized_image,
            background_style,
            (max_size, max_size)
        )
        
        # Вставляем изображение по центру
        paste_x = (max_size - new_width) // 2
        paste_y = (max_size - new_height) // 2
        
        if background_style != BackgroundStyle.TRANSPARENT:
            # Для непрозрачных фонов
            final_image.paste(resized_image, (paste_x, paste_y), resized_image)
        else:
            # Для прозрачного фона
            final_image.paste(resized_image, (paste_x, paste_y))
        
        # Применяем маску если нужно
        if background_style == BackgroundStyle.CIRCLE:
            mask = ImageProcessor.create_circular_mask((max_size, max_size))
            final_image.putalpha(mask)
        elif background_style == BackgroundStyle.ROUNDED:
            mask = ImageProcessor.create_rounded_mask((max_size, max_size), radius=50)
            final_image.putalpha(mask)
        
        # Сохраняем в BytesIO
        output = io.BytesIO()
        
        # Параметры сохранения
        save_kwargs = {
            'format': STICKER_FORMAT,
            'optimize': optimize_size
        }
        
        if STICKER_FORMAT == 'PNG':
            # Для PNG используем сжатие
            save_kwargs['compress_level'] = 9 if optimize_size else 6
        elif STICKER_FORMAT == 'WEBP':
            # Для WebP (если Telegram начнет поддерживать)
            save_kwargs['quality'] = STICKER_QUALITY
            save_kwargs['method'] = 6
        
        await asyncio.to_thread(
            final_image.save,
            output,
            **save_kwargs
        )
        
        output.seek(0)
        
        # Проверяем размер файла
        file_size_kb = len(output.getvalue()) / 1024
        logger.info(f"Стикер обработан: {new_width}x{new_height}, размер: {file_size_kb:.1f} КБ")
        
        # Если файл слишком большой, пробуем дополнительную оптимизацию
        if file_size_kb > 350:  # Telegram ограничение ~350KB для стикеров
            logger.warning(f"Размер стикера слишком большой ({file_size_kb:.1f} КБ), оптимизирую...")
            output = await _optimize_sticker_size(final_image)
        
        return output
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}", exc_info=True)
        return None


async def _create_sticker_background(
    image: Image.Image,
    style: BackgroundStyle,
    size: Tuple[int, int]
) -> Image.Image:
    """
    Создает фон для стикера.
    
    Args:
        image: Изображение стикера
        style: Стиль фона
        size: Размер финального изображения
        
    Returns:
        Изображение с фоном
    """
    if style == BackgroundStyle.TRANSPARENT:
        return Image.new('RGBA', size, (0, 0, 0, 0))
    
    elif style == BackgroundStyle.WHITE:
        return Image.new('RGBA', size, (255, 255, 255, 255))
    
    elif style == BackgroundStyle.GRADIENT:
        # Создаем градиентный фон
        base = Image.new('RGBA', size, (255, 255, 255, 255))
        
        # Создаем градиент
        for y in range(size[1]):
            color_value = int(255 - (y / size[1]) * 50)  # Легкий градиент
            for x in range(size[0]):
                base.putpixel((x, y), (color_value, color_value, 255, 255))
        
        return base
    
    else:
        # Для круглых и закругленных - начинаем с прозрачного
        return Image.new('RGBA', size, (0, 0, 0, 0))


async def _optimize_sticker_size(image: Image.Image) -> io.BytesIO:
    """
    Дополнительно оптимизирует размер стикера.
    
    Args:
        image: Изображение для оптимизации
        
    Returns:
        Оптимизированное изображение в BytesIO
    """
    output = io.BytesIO()
    
    # Пробуем разные уровни качества
    for quality in [95, 90, 85, 80, 75]:
        output.seek(0)
        output.truncate()
        
        if image.mode == 'RGBA':
            # Конвертируем в палитру для уменьшения размера
            converted = image.convert('P', palette=Image.ADAPTIVE, colors=256)
            
            # Сохраняем с оптимизацией
            await asyncio.to_thread(
                converted.save,
                output,
                format='PNG',
                optimize=True,
                compress_level=9
            )
        else:
            await asyncio.to_thread(
                image.save,
                output,
                format='PNG',
                optimize=True,
                compress_level=9,
                quality=quality
            )
        
        output.seek(0)
        size_kb = len(output.getvalue()) / 1024
        
        if size_kb <= 350:
            logger.info(f"Стикер оптимизирован до {size_kb:.1f} КБ")
            break
    
    return output


async def save_image_to_file(
    image_bytes: io.BytesIO,
    file_name: Optional[str] = None,
    storage_dir: str = "storage",
    create_subdirs: bool = True
) -> Optional[str]:
    """
    Сохраняет изображение в файл с улучшенной организацией.
    
    Args:
        image_bytes: Изображение в виде BytesIO
        file_name: Имя файла (если None, генерируется автоматически)
        storage_dir: Директория для сохранения
        create_subdirs: Создавать ли поддиректории по датам
        
    Returns:
        Путь к сохраненному файлу или None при ошибке
    """
    try:
        storage_path = Path(storage_dir)
        
        # Создаем поддиректорию по дате если нужно
        if create_subdirs:
            from datetime import datetime
            date_dir = datetime.now().strftime("%Y-%m-%d")
            storage_path = storage_path / date_dir
        
        # Создаем директорию если не существует
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Генерируем имя файла если не указано
        if not file_name:
            # Генерируем уникальное имя на основе содержимого
            image_bytes.seek(0)
            content_hash = hashlib.md5(image_bytes.read()).hexdigest()[:8]
            image_bytes.seek(0)
            
            file_name = f"sticker_{content_hash}_{uuid.uuid4().hex[:8]}.{STICKER_FORMAT.lower()}"
        
        file_path = storage_path / file_name
        
        # Сохраняем файл
        image_bytes.seek(0)
        await asyncio.to_thread(
            lambda: file_path.write_bytes(image_bytes.read())
        )
        
        # Проверяем, что файл сохранен
        if file_path.exists():
            file_size_kb = file_path.stat().st_size / 1024
            logger.info(f"Изображение сохранено: {file_path} ({file_size_kb:.1f} КБ)")
            return str(file_path)
        else:
            logger.error(f"Файл не был создан: {file_path}")
            return None
            
    except Exception as e:
        logger.error(f"Ошибка при сохранении изображения: {e}", exc_info=True)
        return None


async def cleanup_old_files(storage_dir: str, days: int = 7) -> int:
    """
    Удаляет старые файлы из директории хранения.
    
    Args:
        storage_dir: Директория для очистки
        days: Количество дней для хранения файлов
        
    Returns:
        Количество удаленных файлов
    """
    try:
        from datetime import datetime, timedelta
        
        storage_path = Path(storage_dir)
        if not storage_path.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        # Рекурсивно проходим по всем файлам
        for file_path in storage_path.rglob("*"):
            if file_path.is_file():
                # Проверяем время модификации
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Не удалось удалить файл {file_path}: {e}")
        
        # Удаляем пустые директории
        for dir_path in sorted(storage_path.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception:
                    pass
        
        if deleted_count > 0:
            logger.info(f"Удалено {deleted_count} старых файлов (старше {days} дней)")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Ошибка при очистке старых файлов: {e}")
        return 0


# Дополнительные утилиты
async def get_image_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Получает информацию об изображении.
    
    Args:
        file_path: Путь к файлу изображения
        
    Returns:
        Словарь с информацией или None
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        image = await asyncio.to_thread(Image.open, str(path))
        
        return {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'size_bytes': path.stat().st_size,
            'size_kb': path.stat().st_size / 1024,
            'has_transparency': image.mode in ('RGBA', 'LA') or 
                               (image.mode == 'P' and 'transparency' in image.info)
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения информации об изображении: {e}")
        return None