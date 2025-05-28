import logging
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import asyncio

logger = logging.getLogger(__name__)

# Глобальная переменная для пайплайна Stable Diffusion.
# Загружаем модель один раз при первом вызове.
_pipeline = None

async def load_stable_diffusion_pipeline():
    """Асинхронно загружает пайплайн Stable Diffusion."""
    global _pipeline
    if _pipeline is None:
        logger.info("Загрузка модели Stable Diffusion...")
        # Определяем устройство для загрузки модели: CUDA (Nvidia GPU), MPS (Apple M-чипы), CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Обнаружен Apple Silicon (MPS). Будет использоваться MPS.")
        elif torch.cuda.is_available():
            logger.info("Обнаружен CUDA GPU. Будет использоваться CUDA.")
        else:
            logger.warning("GPU не обнаружен. Модель Stable Diffusion будет работать на CPU. Это может быть очень медленно!")

        # Загрузка Stable Diffusion v1.5
        # Модель загружается из Hugging Face Hub.
        # Для первого запуска потребуется скачать ~4GB.
        # torch_dtype=torch.float16 для GPU (экономит память, быстрее), torch.float32 для CPU.
        # Если на MPS возникают проблемы с float16, попробуйте float32.
        dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

        # Загрузка пайплайна - это блокирующая операция, выполняем ее в отдельном потоке
        _pipeline = await asyncio.to_thread(StableDiffusionPipeline.from_pretrained, 
                                            "runwayml/stable-diffusion-v1-5", 
                                            torch_dtype=dtype)
        
        _pipeline.to(device) # Перемещаем модель на выбранное устройство
        logger.info("Модель Stable Diffusion загружена.")
    return _pipeline

async def generate_sticker_image(prompt: str) -> Image.Image | None:
    """
    Генерирует изображение стикера по текстовому промту с использованием Stable Diffusion.
    """
    logger.info(f"Начинаю генерацию изображения по промту: '{prompt}'")

    pipeline = await load_stable_diffusion_pipeline() # Убеждаемся, что пайплайн загружен

    try:
        # Выполняем генерацию в отдельном потоке, чтобы не блокировать event loop
        # Это важно, так как генерация - это ресурсоемкая и блокирующая операция.
        # Добавляем num_inference_steps=20 (или 30) для ускорения, можно увеличить для лучшего качества.
        image_result = await asyncio.to_thread(lambda: pipeline(prompt, num_inference_steps=20).images[0])
        logger.info(f"Изображение успешно сгенерировано для промта: '{prompt}'")
        return image_result
    except Exception as e:
        logger.error(f"Ошибка при генерации изображения для промта '{prompt}': {e}")
        return None