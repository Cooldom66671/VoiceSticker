"""
Сервис генерации изображений с использованием Stable Diffusion.
Оптимизирован для работы на различных устройствах (CPU, CUDA, MPS).
"""
import asyncio
import time
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import hashlib
import json

import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from logger import get_logger
from config import (
    STABLE_DIFFUSION_MODEL, SD_DEVICE, SD_DTYPE,
    SD_NUM_INFERENCE_STEPS, SD_GUIDANCE_SCALE,
    SD_HEIGHT, SD_WIDTH, SD_TIMEOUT,
    get_device_config
)

logger = get_logger(__name__)

# Глобальные переменные
_pipeline: Optional[StableDiffusionPipeline] = None
_pipeline_lock = asyncio.Lock()
_device_config: Optional[Dict[str, Any]] = None


class GenerationError(Exception):
    """Исключение для ошибок генерации."""
    pass


class Scheduler(Enum):
    """Доступные планировщики для генерации."""
    DPM_SOLVER = "dpm_solver"
    EULER = "euler"
    PNDM = "pndm"


class PromptEnhancer:
    """Класс для улучшения промптов."""
    
    # Стили для стикеров
    STICKER_STYLES = {
        'default': "cute sticker art, kawaii, colorful, simple design, white background",
        'cartoon': "cartoon style sticker, vibrant colors, bold outlines, simple background",
        'anime': "anime style sticker, chibi, kawaii, pastel colors, clean design",
        'minimal': "minimalist sticker design, simple shapes, flat colors, clean",
        'emoji': "emoji style, simple, expressive, bold colors, circular design"
    }
    
    # Негативные промпты для улучшения качества
    NEGATIVE_PROMPTS = {
        'default': "realistic, photo, complex background, text, watermark, signature, low quality, blurry",
        'nsfw': "nsfw, nude, violence, gore, blood, weapons, inappropriate content",
        'quality': "low resolution, pixelated, jpeg artifacts, cropped, out of frame"
    }
    
    @classmethod
    def enhance_prompt(
        cls,
        prompt: str,
        style: str = 'default',
        add_quality_tags: bool = True
    ) -> Tuple[str, str]:
        """
        Улучшает промпт для генерации стикера.
        
        Args:
            prompt: Исходный промпт от пользователя
            style: Стиль стикера
            add_quality_tags: Добавлять ли теги качества
            
        Returns:
            Кортеж (улучшенный промпт, негативный промпт)
        """
        # Базовый промпт
        enhanced = prompt.strip()
        
        # Добавляем стиль
        if style in cls.STICKER_STYLES:
            enhanced = f"{enhanced}, {cls.STICKER_STYLES[style]}"
        
        # Добавляем теги качества
        if add_quality_tags:
            enhanced += ", high quality, 4k, detailed, sharp focus"
        
        # Формируем негативный промпт
        negative_parts = [
            cls.NEGATIVE_PROMPTS['default'],
            cls.NEGATIVE_PROMPTS['nsfw'],
            cls.NEGATIVE_PROMPTS['quality']
        ]
        negative_prompt = ", ".join(negative_parts)
        
        return enhanced, negative_prompt
    
    @classmethod
    def translate_if_needed(cls, prompt: str) -> str:
        """
        Проверяет язык промпта и при необходимости добавляет пояснения.
        (В будущем можно добавить автоперевод)
        
        Args:
            prompt: Исходный промпт
            
        Returns:
            Обработанный промпт
        """
        # Простая проверка на кириллицу
        if any('\u0400' <= char <= '\u04FF' for char in prompt):
            # Промпт на русском - добавляем английские ключевые слова
            logger.debug("Обнаружен русский текст в промпте")
            # В будущем здесь можно добавить автоперевод
        
        return prompt


class PromptCache:
    """Простой кэш для промптов и seed'ов."""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
    
    def get_seed(self, prompt: str) -> Optional[int]:
        """Получает seed для промпта из кэша."""
        return self.cache.get(prompt, {}).get('seed')
    
    def save_seed(self, prompt: str, seed: int):
        """Сохраняет seed для промпта."""
        if len(self.cache) >= self.max_size:
            # Удаляем самый старый элемент
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        
        self.cache[prompt] = {
            'seed': seed,
            'timestamp': time.time()
        }


# Глобальный кэш промптов
_prompt_cache = PromptCache()


async def load_stable_diffusion_pipeline() -> StableDiffusionPipeline:
    """
    Загружает пайплайн Stable Diffusion с оптимизациями.
    
    Returns:
        Загруженный пайплайн
    """
    global _pipeline, _device_config
    
    async with _pipeline_lock:
        if _pipeline is None:
            start_time = time.time()
            logger.info(f"Загрузка Stable Diffusion: {STABLE_DIFFUSION_MODEL}")
            
            try:
                # Получаем конфигурацию устройства
                _device_config = get_device_config()
                device = _device_config['sd_device']
                dtype = _device_config['sd_dtype']
                
                logger.info(f"Устройство: {device}, тип данных: {dtype}")
                
                # Настройки для разных устройств
                pipeline_kwargs = {
                    'torch_dtype': dtype,
                    'use_safetensors': True,  # Более безопасный формат
                    'variant': 'fp16' if dtype == torch.float16 else None,
                }
                
                # Для CPU отключаем safety checker для экономии памяти
                if device == 'cpu':
                    pipeline_kwargs['safety_checker'] = None
                    pipeline_kwargs['requires_safety_checker'] = False
                
                # Загружаем пайплайн
                _pipeline = await asyncio.to_thread(
                    StableDiffusionPipeline.from_pretrained,
                    STABLE_DIFFUSION_MODEL,
                    **pipeline_kwargs
                )
                
                # Перемещаем на устройство
                _pipeline = _pipeline.to(device)
                
                # Оптимизации для разных устройств
                if device == 'cuda':
                    # Включаем оптимизации для CUDA
                    _pipeline.enable_xformers_memory_efficient_attention()
                    _pipeline.enable_vae_slicing()
                    _pipeline.enable_vae_tiling()
                elif device == 'mps':
                    # Оптимизации для Apple Silicon
                    _pipeline.enable_attention_slicing()
                else:
                    # Оптимизации для CPU
                    _pipeline.enable_attention_slicing(slice_size="max")
                    _pipeline.enable_sequential_cpu_offload()
                
                # Используем быстрый планировщик
                _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    _pipeline.scheduler.config
                )
                
                load_time = time.time() - start_time
                logger.info(f"Stable Diffusion загружен за {load_time:.2f} сек")
                
                # Прогреваем модель
                await _warmup_pipeline(_pipeline)
                
            except Exception as e:
                logger.error(f"Ошибка загрузки Stable Diffusion: {e}")
                raise GenerationError(f"Не удалось загрузить модель: {e}")
    
    return _pipeline


async def _warmup_pipeline(pipeline: StableDiffusionPipeline):
    """Прогревает пайплайн для ускорения первой генерации."""
    try:
        logger.debug("Прогрев Stable Diffusion...")
        
        # Генерируем маленькое изображение
        await asyncio.to_thread(
            pipeline,
            "test",
            num_inference_steps=1,
            height=64,
            width=64,
            guidance_scale=1.0
        )
        
        # Очищаем память
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.debug("Пайплайн прогрет")
        
    except Exception as e:
        logger.warning(f"Не удалось прогреть пайплайн: {e}")


async def generate_sticker_image(
    prompt: str,
    style: str = 'default',
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    seed: Optional[int] = None,
    scheduler: Scheduler = Scheduler.DPM_SOLVER,
    enhance_prompt: bool = True
) -> Optional[Image.Image]:
    """
    Генерирует изображение стикера по текстовому промпту.
    
    Args:
        prompt: Текстовое описание
        style: Стиль стикера
        num_inference_steps: Количество шагов генерации
        guidance_scale: Сила следования промпту
        height: Высота изображения
        width: Ширина изображения
        seed: Seed для воспроизводимости
        scheduler: Планировщик для генерации
        enhance_prompt: Улучшать ли промпт автоматически
        
    Returns:
        Сгенерированное изображение или None при ошибке
    """
    start_time = time.time()
    
    # Используем дефолтные значения из конфига
    num_inference_steps = num_inference_steps or SD_NUM_INFERENCE_STEPS
    guidance_scale = guidance_scale or SD_GUIDANCE_SCALE
    height = height or SD_HEIGHT
    width = width or SD_WIDTH
    
    try:
        # Подготавливаем промпт
        if enhance_prompt:
            enhanced_prompt, negative_prompt = PromptEnhancer.enhance_prompt(prompt, style)
            enhanced_prompt = PromptEnhancer.translate_if_needed(enhanced_prompt)
        else:
            enhanced_prompt = prompt
            negative_prompt = ""
        
        logger.info(
            f"Генерация изображения: '{prompt}' "
            f"[стиль: {style}, шаги: {num_inference_steps}, размер: {width}x{height}]"
        )
        logger.debug(f"Улучшенный промпт: '{enhanced_prompt}'")
        
        # Загружаем пайплайн
        pipeline = await load_stable_diffusion_pipeline()
        
        # Настраиваем планировщик
        if scheduler == Scheduler.EULER:
            pipeline.scheduler = EulerDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )
        elif scheduler == Scheduler.PNDM:
            pipeline.scheduler = PNDMScheduler.from_config(
                pipeline.scheduler.config
            )
        
        # Генерируем или используем seed
        if seed is None:
            # Проверяем кэш
            cached_seed = _prompt_cache.get_seed(prompt)
            if cached_seed:
                seed = cached_seed
                logger.debug(f"Используем seed из кэша: {seed}")
            else:
                seed = torch.randint(0, 2**32, (1,)).item()
                _prompt_cache.save_seed(prompt, seed)
        
        generator = torch.Generator(device=_device_config['sd_device']).manual_seed(seed)
        
        # Генерируем изображение с таймаутом
        generate_task = asyncio.create_task(
            asyncio.to_thread(
                pipeline,
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                num_images_per_prompt=1,
                eta=0.0,  # Детерминированная генерация
                callback=None,
                callback_steps=1
            )
        )
        
        result = await asyncio.wait_for(generate_task, timeout=SD_TIMEOUT)
        
        # Получаем изображение
        image = result.images[0]
        
        # Постобработка
        image = await _postprocess_sticker(image)
        
        gen_time = time.time() - start_time
        logger.info(f"Изображение сгенерировано за {gen_time:.2f} сек")
        
        # Очищаем память после генерации
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return image
        
    except asyncio.TimeoutError:
        logger.error(f"Превышен таймаут генерации ({SD_TIMEOUT} сек)")
        return None
        
    except Exception as e:
        logger.error(f"Ошибка при генерации изображения: {e}", exc_info=True)
        return None


async def _postprocess_sticker(image: Image.Image) -> Image.Image:
    """
    Постобработка сгенерированного изображения.
    
    Args:
        image: Исходное изображение
        
    Returns:
        Обработанное изображение
    """
    try:
        # Увеличиваем резкость
        from PIL import ImageEnhance
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # Увеличиваем контрастность
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Увеличиваем насыщенность для стикеров
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)
        
        return image
        
    except Exception as e:
        logger.warning(f"Ошибка постобработки: {e}")
        return image


# Дополнительные утилиты
async def estimate_generation_time(
    num_inference_steps: int,
    device: Optional[str] = None
) -> float:
    """
    Оценивает время генерации.
    
    Args:
        num_inference_steps: Количество шагов
        device: Устройство (если не указано, берется из конфига)
        
    Returns:
        Примерное время в секундах
    """
    if device is None:
        device = _device_config['sd_device'] if _device_config else SD_DEVICE
    
    # Примерные оценки (секунд на шаг)
    time_per_step = {
        'cuda': 0.1,
        'mps': 0.3,
        'cpu': 2.0
    }
    
    base_time = time_per_step.get(device, 1.0)
    return num_inference_steps * base_time + 2.0  # +2 секунды на загрузку


async def get_available_styles() -> List[Dict[str, str]]:
    """
    Возвращает доступные стили стикеров.
    
    Returns:
        Список стилей с описаниями
    """
    return [
        {'id': 'default', 'name': 'Обычный', 'description': 'Милый стикер в стиле kawaii'},
        {'id': 'cartoon', 'name': 'Мультяшный', 'description': 'В стиле мультфильма'},
        {'id': 'anime', 'name': 'Аниме', 'description': 'В стиле аниме/чиби'},
        {'id': 'minimal', 'name': 'Минималистичный', 'description': 'Простой и чистый дизайн'},
        {'id': 'emoji', 'name': 'Эмодзи', 'description': 'В стиле эмодзи'},
    ]