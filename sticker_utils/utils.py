import logging
from PIL import Image
import io
import os
import asyncio

logger = logging.getLogger(__name__)

async def process_image_for_sticker(pil_image: Image.Image) -> io.BytesIO | None:
    """
    Обрабатывает PIL изображение для использования в качестве стикера Telegram.
    Изменяет размер до 512x512, сохраняет пропорции, добавляет прозрачный фон,
    и конвертирует в PNG.
    """
    logger.info("Начинаю обработку изображения для стикера.")
    try:
        max_size = (512, 512)
        original_width, original_height = pil_image.size

        # Создаем новое изображение с прозрачным фоном 512x512
        new_image = Image.new("RGBA", max_size, (0, 0, 0, 0)) # Прозрачный фон

        # Вычисляем новые размеры для вставки изображения с сохранением пропорций
        if original_width > original_height:
            scale_factor = max_size[0] / original_width
            new_width = max_size[0]
            new_height = int(original_height * scale_factor)
        else:
            scale_factor = max_size[1] / original_height
            new_height = max_size[1]
            new_width = int(original_width * scale_factor)

        # Изменяем размер исходного изображения
        # Image.Resampling.LANCZOS - это высококачественный алгоритм ресэмплинга
        resized_image = await asyncio.to_thread(pil_image.resize, (new_width, new_height), Image.Resampling.LANCZOS)

        # Вычисляем позицию для вставки изображения по центру
        paste_x = (max_size[0] - new_width) // 2
        paste_y = (max_size[1] - new_height) // 2

        # Вставляем измененное изображение в новое с прозрачным фоном
        new_image.paste(resized_image, (paste_x, paste_y))

        # Сохраняем изображение в BytesIO объект как PNG
        byte_arr = io.BytesIO()
        # new_image.save - это блокирующая операция, выполняем ее в отдельном потоке
        await asyncio.to_thread(new_image.save, byte_arr, format='PNG')
        byte_arr.seek(0) # Перематываем указатель в начало
        logger.info("Изображение успешно обработано для стикера.")
        return byte_arr
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения для стикера: {e}")
        return None

async def save_image_to_file(image_bytes: io.BytesIO, file_name: str, storage_dir: str) -> str | None:
    """
    Сохраняет BytesIO изображение в файл.
    """
    file_path = os.path.join(storage_dir, file_name)
    try:
        # Операция записи в файл - это блокирующая операция, выполняем ее в отдельном потоке
        await asyncio.to_thread(lambda: open(file_path, 'wb').write(image_bytes.read()))
        logger.info(f"Изображение успешно сохранено в: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Ошибка при сохранении изображения в файл {file_path}: {e}")
        return None