"""
Advanced image generation service using Stable Diffusion.
Optimized for multiple devices with caching, safety checks, and quality enhancements.
"""
import asyncio
import time
import gc
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import platform
import torch
import numpy as np
from PIL import Image
import io

from logger import get_logger

logger = get_logger(__name__)


# Custom exception for generation errors
class GenerationError(Exception):
    """Custom exception for image generation errors."""
    pass

# Try to import diffusers, but handle if not available
try:
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        PNDMScheduler
    )
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor
    DIFFUSERS_AVAILABLE = True
except ImportError:
    logger.warning("diffusers not available. Image generation will be limited.")
    DIFFUSERS_AVAILABLE = False


# Global variables for model and stats
_pipeline: Optional[Any] = None
_device: Optional[str] = None
_generation_stats = {
    'total_generations': 0,
    'successful_generations': 0,
    'failed_generations': 0,
    'total_time': 0.0,
    'last_generation': None,
    'errors': []
}


def _check_hardware_capabilities() -> Tuple[bool, str]:
    """Check if hardware can run Stable Diffusion."""
    try:
        if torch.cuda.is_available():
            # Check CUDA memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 4:
                return False, f"Insufficient GPU memory: {gpu_memory:.1f}GB (need 4GB+)"
            return True, "cuda"
        elif torch.backends.mps.is_available():
            # Check for Apple Silicon
            return True, "mps"
        else:
            # Check CPU and RAM
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb < 8:
                return False, f"Insufficient RAM: {ram_gb:.1f}GB (need 8GB+)"
            return True, "cpu"
    except Exception as e:
        logger.error(f"Hardware check failed: {e}")
        return False, str(e)


async def load_stable_diffusion_pipeline():
    """Load Stable Diffusion pipeline with optimizations.
    
    Raises:
        GenerationError: If critical loading error occurs in production
    """
    global _pipeline, _device
    
    if _pipeline is not None:
        logger.info("Stable Diffusion pipeline already loaded")
        return
    
    # Check if we should skip loading (for testing)
    if os.getenv('SKIP_SD_LOADING', '').lower() == 'true':
        logger.info("Skipping Stable Diffusion loading (SKIP_SD_LOADING=true)")
        _pipeline = "placeholder"
        return
    
    # Check hardware capabilities
    can_run, device_or_error = _check_hardware_capabilities()
    if not can_run:
        logger.warning(f"Cannot run Stable Diffusion: {device_or_error}")
        logger.info("Using placeholder image generator")
        _pipeline = "placeholder"
        if os.getenv('REQUIRE_SD', '').lower() == 'true':
            raise GenerationError(f"Stable Diffusion required but cannot run: {device_or_error}")
        return
    
    if not DIFFUSERS_AVAILABLE:
        logger.warning("diffusers library not installed. Using placeholder generator.")
        _pipeline = "placeholder"
        if os.getenv('REQUIRE_SD', '').lower() == 'true':
            raise GenerationError("Stable Diffusion required but diffusers not installed")
        return
    
    _device = device_or_error
    logger.info(f"Loading Stable Diffusion pipeline on {_device}...")
    
    try:
        # Model configuration
        model_id = os.getenv('SD_MODEL_ID', 'stabilityai/stable-diffusion-2-1-base')
        cache_dir = os.getenv('SD_CACHE_DIR', './models')
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load pipeline with optimizations
        start_time = time.time()
        
        # Load with appropriate precision
        if _device == "cuda":
            _pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                use_safetensors=True,
                safety_checker=None  # Disable for performance
            )
        else:
            _pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                use_safetensors=True,
                safety_checker=None
            )
        
        # Move to device
        _pipeline = _pipeline.to(_device)
        
        # Enable optimizations
        if _device == "cuda":
            _pipeline.enable_attention_slicing()
            _pipeline.enable_vae_slicing()
            
            # Try to enable xformers for memory efficiency
            try:
                _pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✓ xformers enabled for memory efficiency")
            except Exception:
                logger.debug("xformers not available")
        
        # Use faster scheduler
        _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            _pipeline.scheduler.config
        )
        
        # Warmup run
        logger.info("Running warmup generation...")
        with torch.no_grad():
            _ = _pipeline(
                "test",
                num_inference_steps=1,
                width=64,
                height=64,
                output_type="pil"
            ).images[0]
        
        load_time = time.time() - start_time
        logger.info(f"✓ Stable Diffusion loaded successfully in {load_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Failed to load Stable Diffusion: {e}")
        logger.info("Falling back to placeholder generator")
        _pipeline = "placeholder"


async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    style: Optional[str] = None
) -> Optional[bytes]:
    """Generate image from text prompt.
    
    Raises:
        GenerationError: If image generation fails
    """
    global _generation_stats
    
    if not prompt or len(prompt.strip()) == 0:
        raise GenerationError("Prompt cannot be empty")
    
    # Validate dimensions
    if width <= 0 or height <= 0 or width > 1024 or height > 1024:
        raise GenerationError(f"Invalid dimensions: {width}x{height}")
    
    start_time = time.time()
    _generation_stats['total_generations'] += 1
    
    try:
        # Apply style presets
        styled_prompt, styled_negative = _apply_style_preset(
            prompt, negative_prompt, style
        )
        
        # Generate image
        if _pipeline == "placeholder" or not DIFFUSERS_AVAILABLE:
            # Use placeholder generator
            image = await _generate_placeholder_image(
                styled_prompt, width, height
            )
        else:
            # Use real Stable Diffusion
            image = await _generate_with_sd(
                styled_prompt,
                styled_negative,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                seed
            )
        
        if image is None:
            raise GenerationError("Failed to generate image")
        
        # Convert to bytes
        bio = io.BytesIO()
        image.save(bio, format='PNG', optimize=True)
        image_bytes = bio.getvalue()
        
        # Update stats
        generation_time = time.time() - start_time
        _generation_stats['successful_generations'] += 1
        _generation_stats['total_time'] += generation_time
        _generation_stats['last_generation'] = {
            'prompt': prompt,
            'style': style,
            'size': f"{width}x{height}",
            'time': generation_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"Image generated successfully: {len(image_bytes)/1024:.1f}KB, "
            f"time: {generation_time:.1f}s"
        )
        
        return image_bytes
        
    except GenerationError:
        raise
    except Exception as e:
        _generation_stats['failed_generations'] += 1
        _generation_stats['errors'].append({
            'error': str(e),
            'prompt': prompt[:50],
            'timestamp': datetime.now().isoformat()
        })
        logger.error(f"Image generation failed: {e}")
        raise GenerationError(f"Generation failed: {str(e)}")


async def _generate_placeholder_image(
    prompt: str,
    width: int,
    height: int
) -> Optional[Image.Image]:
    """Generate a simple placeholder image."""
    try:
        # Create a gradient background
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        # Simple gradient based on prompt hash
        prompt_hash = hash(prompt) % 360
        
        for y in range(height):
            for x in range(width):
                # Create HSV color
                h = (prompt_hash + x / width * 60) % 360
                s = 0.7 + (y / height) * 0.3
                v = 0.8 + (1 - y / height) * 0.2
                
                # Convert HSV to RGB
                c = v * s
                x_val = c * (1 - abs((h / 60) % 2 - 1))
                m = v - c
                
                if h < 60:
                    r, g, b = c, x_val, 0
                elif h < 120:
                    r, g, b = x_val, c, 0
                elif h < 180:
                    r, g, b = 0, c, x_val
                elif h < 240:
                    r, g, b = 0, x_val, c
                elif h < 300:
                    r, g, b = x_val, 0, c
                else:
                    r, g, b = c, 0, x_val
                
                pixels[x, y] = (
                    int((r + m) * 255),
                    int((g + m) * 255),
                    int((b + m) * 255)
                )
        
        # Add text overlay
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", size=min(width, height) // 20)
            except:
                font = ImageFont.load_default()
            
            # Draw prompt text
            text = f"Generated: {prompt[:50]}..."
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = (width - text_width) // 2
            text_y = height - text_height - 20
            
            # Draw text with shadow
            shadow_offset = 2
            draw.text(
                (text_x + shadow_offset, text_y + shadow_offset),
                text,
                fill=(0, 0, 0, 128),
                font=font
            )
            draw.text(
                (text_x, text_y),
                text,
                fill=(255, 255, 255),
                font=font
            )
            
        except Exception as e:
            logger.debug(f"Could not add text overlay: {e}")
        
        logger.info("Generated placeholder image")
        return image
        
    except Exception as e:
        logger.error(f"Failed to generate placeholder: {e}")
        return None


async def _generate_with_sd(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int]
) -> Optional[Image.Image]:
    """Generate image using Stable Diffusion."""
    try:
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=_device).manual_seed(seed)
        
        # Run generation in executor to not block
        loop = asyncio.get_event_loop()
        
        def _generate():
            with torch.no_grad():
                result = _pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="pil"
                )
                return result.images[0]
        
        image = await loop.run_in_executor(None, _generate)
        
        # Clear memory
        if _device == "cuda":
            torch.cuda.empty_cache()
        
        return image
        
    except Exception as e:
        logger.error(f"SD generation failed: {e}")
        return None


def _apply_style_preset(
    prompt: str,
    negative_prompt: Optional[str],
    style: Optional[str]
) -> Tuple[str, str]:
    """Apply style presets to prompts."""
    
    # Default negative prompt
    if negative_prompt is None:
        negative_prompt = (
            "low quality, blurry, pixelated, noisy, watermark, text, "
            "signature, username, artist name, bad anatomy, bad proportions"
        )
    
    # Style presets
    style_presets = {
        'anime': {
            'prefix': 'anime style, anime art, cel shading, ',
            'suffix': ', trending on pixiv, anime key visual',
            'negative': ', realistic, photorealistic, 3d render'
        },
        'realistic': {
            'prefix': 'photorealistic, highly detailed, professional photography, ',
            'suffix': ', 8k uhd, dslr, high quality',
            'negative': ', anime, cartoon, drawing, illustration'
        },
        'artistic': {
            'prefix': 'artistic, oil painting style, masterpiece, ',
            'suffix': ', artstation, by greg rutkowski',
            'negative': ', photo, realistic'
        },
        'cartoon': {
            'prefix': 'cartoon style, colorful, vibrant, ',
            'suffix': ', pixar style, 3d animation',
            'negative': ', realistic, photo'
        },
        'sketch': {
            'prefix': 'pencil sketch, black and white drawing, ',
            'suffix': ', detailed linework',
            'negative': ', color, painted, photograph'
        }
    }
    
    # Apply style if specified
    if style and style.lower() in style_presets:
        preset = style_presets[style.lower()]
        prompt = preset['prefix'] + prompt + preset['suffix']
        negative_prompt = negative_prompt + preset['negative']
    
    return prompt, negative_prompt


def get_available_styles() -> List[Dict[str, str]]:
    """Get list of available style presets.
    
    Returns:
        List of style dictionaries with name and description
    """
    return [
        {
            'name': 'anime',
            'description': '🎌 Anime/Manga style with cel shading',
            'emoji': '🎌'
        },
        {
            'name': 'realistic',
            'description': '📸 Photorealistic, high-quality photography',
            'emoji': '📸'
        },
        {
            'name': 'artistic',
            'description': '🎨 Artistic oil painting style',
            'emoji': '🎨'
        },
        {
            'name': 'cartoon',
            'description': '🎭 Colorful cartoon/Pixar style',
            'emoji': '🎭'
        },
        {
            'name': 'sketch',
            'description': '✏️ Black and white pencil sketch',
            'emoji': '✏️'
        }
    ]


def estimate_generation_time(
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 25,
    is_sticker: bool = False
) -> float:
    """Estimate generation time in seconds.
    
    Args:
        width: Image width
        height: Image height
        num_inference_steps: Number of denoising steps
        is_sticker: Whether generating a sticker (usually faster)
    
    Returns:
        Estimated time in seconds
    """
    # Base time estimates
    if _pipeline == "placeholder" or not DIFFUSERS_AVAILABLE:
        # Placeholder generator is very fast
        return 0.5
    
    # Calculate based on resolution and steps
    pixels = width * height
    base_pixels = 512 * 512  # Standard resolution
    
    # Time scaling factors
    resolution_factor = pixels / base_pixels
    
    # Device-specific base times (for 512x512, 25 steps)
    if _device == "cuda":
        base_time = 3.0  # Fast GPU
    elif _device == "mps":
        base_time = 5.0  # Apple Silicon
    else:
        base_time = 15.0  # CPU is slow
    
    # Adjust for steps (linear scaling)
    steps_factor = num_inference_steps / 25.0
    
    # Stickers are typically faster (fewer steps)
    if is_sticker:
        steps_factor *= 0.8
    
    # Calculate total time
    estimated_time = base_time * resolution_factor * steps_factor
    
    # Add some buffer for overhead
    estimated_time *= 1.2
    
    # Round to 1 decimal place
    return round(estimated_time, 1)


async def get_generation_stats() -> Dict[str, Any]:
    """Get image generation statistics."""
    stats = _generation_stats.copy()
    
    # Calculate averages
    if stats['successful_generations'] > 0:
        stats['average_generation_time'] = (
            stats['total_time'] / stats['successful_generations']
        )
    else:
        stats['average_generation_time'] = 0
    
    # Device info
    stats['device'] = _device or 'not loaded'
    stats['model_loaded'] = _pipeline is not None
    
    # Memory usage
    if _device == "cuda" and torch.cuda.is_available():
        stats['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)
        stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024**3)
    
    # Keep only last 10 errors
    if len(stats['errors']) > 10:
        stats['errors'] = stats['errors'][-10:]
    
    return stats


async def generate_sticker_image(
    prompt: str,
    style: str = "cartoon",
    negative_prompt: Optional[str] = None
) -> Optional[bytes]:
    """Generate sticker-optimized image.
    
    Stickers have specific requirements:
    - Square format (512x512)
    - Clear, simple subjects
    - High contrast
    - Cartoon/anime style works best
    
    Raises:
        GenerationError: If sticker generation fails
    """
    logger.info(f"Generating sticker image: {prompt[:50]}...")
    
    # Enhance prompt for sticker generation
    sticker_prompt = f"sticker design, {prompt}, simple background, centered composition"
    
    # Enhanced negative prompt for stickers
    sticker_negative = (
        "complex background, multiple subjects, text, watermark, "
        "blurry, low contrast, photorealistic"
    )
    
    if negative_prompt:
        sticker_negative = f"{sticker_negative}, {negative_prompt}"
    
    # Generate with sticker-optimized parameters
    try:
        return await generate_image(
            prompt=sticker_prompt,
            negative_prompt=sticker_negative,
            width=512,
            height=512,
            num_inference_steps=20,  # Fewer steps for faster generation
            guidance_scale=7.5,
            style=style
        )
    except GenerationError as e:
        logger.error(f"Sticker generation failed: {e}")
        raise


async def clear_cache():
    """Clear image generation cache and free memory."""
    logger.info("Clearing image generation cache...")
    
    try:
        # Clear CUDA cache if available
        if _device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("✓ CUDA cache cleared")
        
        # Run garbage collection
        gc.collect()
        logger.info("✓ Garbage collection completed")
        
        # In the future, you can add:
        # - Clear any disk-based image cache
        # - Clear prompt embeddings cache
        # - Clear safety checker cache
        
        logger.info("Image generation cache cleared successfully")
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")


# Optional: Add function to unload model to free memory
async def unload_model():
    """Unload the model to free memory."""
    global _pipeline
    
    if _pipeline is not None and _pipeline != "placeholder":
        logger.info("Unloading Stable Diffusion model...")
        
        try:
            del _pipeline
            _pipeline = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("✓ Model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")