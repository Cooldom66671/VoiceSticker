"""
Advanced sticker processing utilities with optimizations for Telegram.
Fixes black screen issues and provides professional image processing.
"""
import asyncio
import io
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime, timedelta

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageEnhance
import cv2
import rembg
from colorthief import ColorThief

from logger import get_logger, log_execution_time, log_error
from config import config

logger = get_logger(__name__)


class StickerError(Exception):
    """Base exception for sticker processing errors."""
    pass


class ProcessingError(StickerError):
    """Image processing failed."""
    pass


class BackgroundStyle(Enum):
    """Available background styles for stickers."""
    TRANSPARENT = "transparent"
    WHITE = "white"
    BLACK = "black"
    GRADIENT = "gradient"
    GRADIENT_RADIAL = "gradient_radial"
    CIRCLE = "circle"
    ROUNDED = "rounded"
    HEXAGON = "hexagon"
    STAR = "star"
    HEART = "heart"
    CUSTOM_COLOR = "custom_color"
    BLUR = "blur"
    PATTERN = "pattern"


@dataclass
class ProcessingOptions:
    """Options for sticker processing."""
    background_style: BackgroundStyle = BackgroundStyle.TRANSPARENT
    background_color: Optional[Tuple[int, int, int, int]] = None
    add_shadow: bool = True
    shadow_offset: Tuple[int, int] = (5, 5)
    shadow_blur: int = 10
    shadow_opacity: int = 128
    shadow_color: Tuple[int, int, int] = (0, 0, 0)
    add_outline: bool = False
    outline_size: int = 3
    outline_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    remove_background: bool = True
    background_threshold: int = 240
    optimize_size: bool = True
    target_size: int = 512
    add_padding: bool = True
    padding_percent: float = 0.05
    enhance_colors: bool = True
    auto_crop: bool = True
    maintain_aspect_ratio: bool = True


class BackgroundRemover:
    """Advanced background removal using multiple methods."""
    
    def __init__(self):
        """Initialize background remover."""
        self._rembg_session = None
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    async def remove_background(
        self,
        image: Image.Image,
        method: str = "auto"
    ) -> Image.Image:
        """
        Remove background from image.
        
        Args:
            image: Input image
            method: Removal method ('auto', 'rembg', 'color', 'alpha')
            
        Returns:
            Image with transparent background
        """
        if method == "auto":
            # Try multiple methods and pick best result
            results = []
            
            # Try rembg if available
            try:
                result = await self._remove_with_rembg(image)
                results.append(('rembg', result))
            except Exception as e:
                logger.debug(f"Rembg failed: {e}")
            
            # Try color-based removal
            try:
                result = await self._remove_by_color(image)
                results.append(('color', result))
            except Exception as e:
                logger.debug(f"Color removal failed: {e}")
            
            # Pick best result based on transparency
            if results:
                best_method, best_result = max(
                    results,
                    key=lambda x: self._calculate_transparency_score(x[1])
                )
                logger.debug(f"Best background removal method: {best_method}")
                return best_result
            
            # Fallback to simple alpha
            return await self._remove_by_alpha(image)
        
        elif method == "rembg":
            return await self._remove_with_rembg(image)
        elif method == "color":
            return await self._remove_by_color(image)
        else:
            return await self._remove_by_alpha(image)
    
    async def _remove_with_rembg(self, image: Image.Image) -> Image.Image:
        """Remove background using rembg AI model."""
        if self._rembg_session is None:
            # Lazy load rembg
            self._rembg_session = rembg.new_session('u2net')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Process with rembg
        output_bytes = await asyncio.to_thread(
            rembg.remove,
            img_bytes.getvalue(),
            session=self._rembg_session
        )
        
        # Convert back to PIL Image
        return Image.open(io.BytesIO(output_bytes))
    
    async def _remove_by_color(
        self,
        image: Image.Image,
        threshold: int = 240,
        tolerance: int = 20
    ) -> Image.Image:
        """Remove background by color threshold."""
        # Convert to RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array
        data = np.array(image)
        
        # Find dominant background color (usually in corners)
        corner_colors = [
            data[0, 0, :3],  # Top-left
            data[0, -1, :3],  # Top-right
            data[-1, 0, :3],  # Bottom-left
            data[-1, -1, :3]  # Bottom-right
        ]
        
        # Calculate mean background color
        bg_color = np.mean(corner_colors, axis=0).astype(int)
        
        # Create mask for background color
        color_distance = np.sqrt(np.sum((data[:, :, :3] - bg_color) ** 2, axis=2))
        mask = color_distance < tolerance
        
        # Also check for very bright colors (white background)
        bright_mask = np.all(data[:, :, :3] > threshold, axis=2)
        
        # Combine masks
        final_mask = mask | bright_mask
        
        # Apply mask to alpha channel
        data[:, :, 3] = np.where(final_mask, 0, 255)
        
        # Smooth edges
        alpha_channel = data[:, :, 3]
        alpha_channel = await asyncio.to_thread(
            cv2.morphologyEx,
            alpha_channel,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )
        alpha_channel = await asyncio.to_thread(
            cv2.GaussianBlur,
            alpha_channel,
            (5, 5),
            0
        )
        data[:, :, 3] = alpha_channel
        
        return Image.fromarray(data, 'RGBA')
    
    async def _remove_by_alpha(self, image: Image.Image) -> Image.Image:
        """Simple alpha-based background removal."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get alpha channel
        alpha = image.split()[3]
        
        # Create binary mask
        mask = Image.eval(alpha, lambda a: 255 if a > 128 else 0)
        
        # Apply mask
        image.putalpha(mask)
        
        return image
    
    def _calculate_transparency_score(self, image: Image.Image) -> float:
        """Calculate how much of the image is transparent."""
        if image.mode != 'RGBA':
            return 0.0
        
        alpha = np.array(image.split()[3])
        transparent_pixels = np.sum(alpha < 128)
        total_pixels = alpha.size
        
        return transparent_pixels / total_pixels


class ImageEffects:
    """Advanced image effects and filters."""
    
    @staticmethod
    async def add_shadow(
        image: Image.Image,
        offset: Tuple[int, int] = (5, 5),
        blur_radius: int = 10,
        opacity: int = 128,
        color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Image.Image:
        """Add drop shadow to image."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Calculate canvas size with shadow
        shadow_padding = max(abs(offset[0]), abs(offset[1])) + blur_radius * 2
        new_width = image.width + shadow_padding * 2
        new_height = image.height + shadow_padding * 2
        
        # Create canvas
        canvas = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
        
        # Create shadow
        shadow = Image.new('RGBA', image.size, (*color, 0))
        
        # Get alpha channel for shadow shape
        _, _, _, alpha = image.split()
        
        # Create shadow with opacity
        shadow_alpha = Image.eval(alpha, lambda a: min(int(a * opacity / 255), 255))
        shadow.putalpha(shadow_alpha)
        
        # Blur shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Paste shadow with offset
        shadow_pos = (
            shadow_padding + offset[0],
            shadow_padding + offset[1]
        )
        canvas.paste(shadow, shadow_pos)
        
        # Paste original image
        image_pos = (shadow_padding, shadow_padding)
        canvas.paste(image, image_pos, image)
        
        return canvas
    
    @staticmethod
    async def add_outline(
        image: Image.Image,
        size: int = 3,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        style: str = "solid"
    ) -> Image.Image:
        """Add outline to image."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get alpha channel
        _, _, _, alpha = image.split()
        
        if style == "solid":
            # Create dilated mask for outline
            outline_mask = alpha.filter(ImageFilter.MaxFilter(size * 2 + 1))
            
            # Create outline image
            outline = Image.new('RGBA', image.size, color)
            outline.putalpha(outline_mask)
            
            # Composite with original
            result = Image.new('RGBA', image.size, (0, 0, 0, 0))
            result.paste(outline, (0, 0))
            result.paste(image, (0, 0), image)
            
        elif style == "glow":
            # Create multiple layers for glow effect
            result = image.copy()
            
            for i in range(size, 0, -1):
                glow_mask = alpha.filter(ImageFilter.MaxFilter(i * 2 + 1))
                glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(i))
                
                glow_layer = Image.new('RGBA', image.size, color)
                glow_opacity = int(255 * (i / size) * 0.5)
                glow_alpha = Image.eval(glow_mask, lambda a: min(a, glow_opacity))
                glow_layer.putalpha(glow_alpha)
                
                result = Image.alpha_composite(glow_layer, result)
        
        else:  # gradient
            # Create gradient outline
            result = Image.new('RGBA', image.size, (0, 0, 0, 0))
            
            for i in range(size, 0, -1):
                layer_mask = alpha.filter(ImageFilter.MaxFilter(i * 2 + 1))
                layer_color = tuple(
                    int(c * (i / size)) 
                    for c in color[:3]
                ) + (color[3],)
                
                layer = Image.new('RGBA', image.size, layer_color)
                layer.putalpha(layer_mask)
                result = Image.alpha_composite(result, layer)
            
            result.paste(image, (0, 0), image)
        
        return result
    
    @staticmethod
    async def add_background(
        image: Image.Image,
        style: BackgroundStyle,
        color: Optional[Tuple[int, int, int]] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Add background to image."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Determine target size
        if size is None:
            size = (config.sticker.max_size, config.sticker.max_size)
        
        # Create background
        if style == BackgroundStyle.TRANSPARENT:
            background = Image.new('RGBA', size, (0, 0, 0, 0))
            
        elif style == BackgroundStyle.WHITE:
            background = Image.new('RGBA', size, (255, 255, 255, 255))
            
        elif style == BackgroundStyle.BLACK:
            background = Image.new('RGBA', size, (0, 0, 0, 255))
            
        elif style == BackgroundStyle.CUSTOM_COLOR:
            if color is None:
                color = (255, 255, 255)
            background = Image.new('RGBA', size, (*color, 255))
            
        elif style == BackgroundStyle.GRADIENT:
            background = await ImageEffects._create_gradient(
                size, 
                direction="vertical"
            )
            
        elif style == BackgroundStyle.GRADIENT_RADIAL:
            background = await ImageEffects._create_gradient(
                size,
                direction="radial"
            )
            
        elif style == BackgroundStyle.BLUR:
            # Create blurred version of image as background
            background = image.resize(size, Image.Resampling.LANCZOS)
            background = background.filter(ImageFilter.GaussianBlur(30))
            enhancer = ImageEnhance.Brightness(background)
            background = enhancer.enhance(1.2)
            
        elif style == BackgroundStyle.PATTERN:
            background = await ImageEffects._create_pattern(size)
            
        else:
            # For shaped backgrounds, start with transparent
            background = Image.new('RGBA', size, (0, 0, 0, 0))
        
        # Apply shape mask if needed
        if style in [BackgroundStyle.CIRCLE, BackgroundStyle.ROUNDED, 
                    BackgroundStyle.HEXAGON, BackgroundStyle.STAR, 
                    BackgroundStyle.HEART]:
            
            mask = await ImageEffects._create_shape_mask(size, style)
            
            # If not transparent style, apply mask to colored background
            if style != BackgroundStyle.TRANSPARENT:
                white_bg = Image.new('RGBA', size, (255, 255, 255, 255))
                background = Image.composite(white_bg, background, mask)
        
        # Calculate position to center image
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        
        # Paste image onto background
        if image.mode == 'RGBA':
            background.paste(image, (x, y), image)
        else:
            background.paste(image, (x, y))
        
        # Apply shape mask to final image if needed
        if style in [BackgroundStyle.CIRCLE, BackgroundStyle.ROUNDED,
                    BackgroundStyle.HEXAGON, BackgroundStyle.STAR,
                    BackgroundStyle.HEART]:
            
            # Create final image with mask
            final = Image.new('RGBA', size, (0, 0, 0, 0))
            final.paste(background, (0, 0), mask)
            return final
        
        return background
    
    @staticmethod
    async def _create_gradient(
        size: Tuple[int, int],
        direction: str = "vertical",
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> Image.Image:
        """Create gradient background."""
        width, height = size
        
        if colors is None:
            # Default gradient colors
            colors = [(255, 200, 200), (200, 200, 255)]
        
        gradient = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient)
        
        if direction == "vertical":
            for y in range(height):
                # Interpolate color
                ratio = y / height
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                
                draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
                
        elif direction == "horizontal":
            for x in range(width):
                ratio = x / width
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                
                draw.line([(x, 0), (x, height)], fill=(r, g, b, 255))
                
        elif direction == "radial":
            center_x, center_y = width // 2, height // 2
            max_radius = np.sqrt(center_x**2 + center_y**2)
            
            for y in range(height):
                for x in range(width):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    ratio = min(distance / max_radius, 1.0)
                    
                    r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                    g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                    b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                    
                    gradient.putpixel((x, y), (r, g, b, 255))
        
        return gradient
    
    @staticmethod
    async def _create_pattern(size: Tuple[int, int]) -> Image.Image:
        """Create pattern background."""
        pattern = Image.new('RGBA', size, (240, 240, 240, 255))
        draw = ImageDraw.Draw(pattern)
        
        # Draw simple dot pattern
        spacing = 20
        dot_size = 3
        
        for y in range(0, size[1], spacing):
            for x in range(0, size[0], spacing):
                draw.ellipse(
                    [x - dot_size, y - dot_size, x + dot_size, y + dot_size],
                    fill=(220, 220, 220, 255)
                )
        
        return pattern
    
    @staticmethod
    async def _create_shape_mask(
        size: Tuple[int, int],
        style: BackgroundStyle
    ) -> Image.Image:
        """Create shape mask for background."""
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = size
        
        if style == BackgroundStyle.CIRCLE:
            # Draw circle
            draw.ellipse([0, 0, width, height], fill=255)
            
        elif style == BackgroundStyle.ROUNDED:
            # Draw rounded rectangle
            radius = min(width, height) // 10
            draw.rounded_rectangle(
                [0, 0, width, height],
                radius=radius,
                fill=255
            )
            
        elif style == BackgroundStyle.HEXAGON:
            # Draw hexagon
            cx, cy = width // 2, height // 2
            r = min(width, height) // 2 - 10
            
            points = []
            for i in range(6):
                angle = i * np.pi / 3
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                points.append((x, y))
            
            draw.polygon(points, fill=255)
            
        elif style == BackgroundStyle.STAR:
            # Draw star
            cx, cy = width // 2, height // 2
            outer_r = min(width, height) // 2 - 10
            inner_r = outer_r // 2
            
            points = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi / 2
                r = outer_r if i % 2 == 0 else inner_r
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                points.append((x, y))
            
            draw.polygon(points, fill=255)
            
        elif style == BackgroundStyle.HEART:
            # Draw heart shape
            cx, cy = width // 2, height // 2
            scale = min(width, height) / 200
            
            # Heart parametric equations
            points = []
            for t in np.linspace(0, 2 * np.pi, 100):
                x = 16 * (np.sin(t) ** 3)
                y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
                
                x = cx + x * scale
                y = cy + y * scale
                points.append((x, y))
            
            draw.polygon(points, fill=255)
        
        # Smooth edges
        mask = mask.filter(ImageFilter.GaussianBlur(2))
        
        return mask


class StickerProcessor:
    """Main sticker processing class."""
    
    def __init__(self):
        """Initialize processor."""
        self.bg_remover = BackgroundRemover()
        self.effects = ImageEffects()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @log_execution_time
    @log_error("sticker_processing")
    async def process_image(
        self,
        image: Image.Image,
        options: Optional[ProcessingOptions] = None
    ) -> io.BytesIO:
        """
        Process image for use as Telegram sticker.
        
        Args:
            image: Input image
            options: Processing options
            
        Returns:
            BytesIO with processed image
        """
        if options is None:
            options = ProcessingOptions()
        
        logger.info(f"Processing image for sticker [background: {options.background_style.value}]")
        
        try:
            # Step 1: Convert to RGBA
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
                logger.debug("Converted to RGBA")
            
            # Step 2: Remove background if needed
            if options.remove_background and options.background_style == BackgroundStyle.TRANSPARENT:
                image = await self.bg_remover.remove_background(image)
                logger.debug("Background removed")
            
            # Step 3: Auto-crop if enabled
            if options.auto_crop:
                image = await self._auto_crop(image)
                logger.debug("Auto-cropped")
            
            # Step 4: Resize to target size
            image = await self._resize_with_padding(
                image,
                options.target_size,
                options.maintain_aspect_ratio,
                options.padding_percent if options.add_padding else 0
            )
            logger.debug(f"Resized to {image.size}")
            
            # Step 5: Apply effects
            if options.add_outline:
                image = await self.effects.add_outline(
                    image,
                    size=options.outline_size,
                    color=options.outline_color,
                    style="glow" if options.background_style == BackgroundStyle.TRANSPARENT else "solid"
                )
                logger.debug("Outline added")
            
            if options.add_shadow and options.background_style == BackgroundStyle.TRANSPARENT:
                image = await self.effects.add_shadow(
                    image,
                    offset=options.shadow_offset,
                    blur_radius=options.shadow_blur,
                    opacity=options.shadow_opacity,
                    color=options.shadow_color
                )
                logger.debug("Shadow added")
            
            # Step 6: Apply background
            image = await self.effects.add_background(
                image,
                style=options.background_style,
                color=options.background_color,
                size=(options.target_size, options.target_size)
            )
            logger.debug(f"Background applied: {options.background_style.value}")
            
            # Step 7: Enhance colors if enabled
            if options.enhance_colors:
                image = await self._enhance_image(image)
                logger.debug("Colors enhanced")
            
            # Step 8: Save to BytesIO
            output = await self._save_optimized(image, options.optimize_size)
            
            # Log final stats
            file_size_kb = len(output.getvalue()) / 1024
            logger.info(f"Sticker processed: {image.size}, {file_size_kb:.1f} KB")
            
            return output
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise ProcessingError(f"Failed to process image: {e}")
    
    async def _auto_crop(self, image: Image.Image) -> Image.Image:
        """Auto-crop transparent borders."""
        if image.mode != 'RGBA':
            return image
        
        # Get bounding box of non-transparent pixels
        bbox = image.getbbox()
        
        if bbox:
            # Add small margin
            margin = 5
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.width, x2 + margin)
            y2 = min(image.height, y2 + margin)
            
            return image.crop((x1, y1, x2, y2))
        
        return image
    
    async def _resize_with_padding(
        self,
        image: Image.Image,
        target_size: int,
        maintain_aspect: bool = True,
        padding_percent: float = 0.05
    ) -> Image.Image:
        """Resize image with padding to maintain aspect ratio."""
        if not maintain_aspect:
            return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Calculate target size with padding
        padding = int(target_size * padding_percent)
        actual_size = target_size - (padding * 2)
        
        # Calculate scaling factor
        scale = min(actual_size / image.width, actual_size / image.height)
        
        # Don't upscale if image is smaller
        if scale > 1.0:
            scale = 1.0
        
        # Calculate new size
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create canvas with padding
        canvas = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
        
        # Calculate position to center image
        x = (target_size - new_width) // 2
        y = (target_size - new_height) // 2
        
        # Paste resized image
        canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)
        
        return canvas
    
    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image colors and sharpness."""
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.15)
        
        return image
    
    async def _save_optimized(
        self,
        image: Image.Image,
        optimize: bool = True
    ) -> io.BytesIO:
        """Save image with optimization."""
        output = io.BytesIO()
        
        # Prepare save parameters
        save_params = {
            'format': config.sticker.format,
            'optimize': optimize
        }
        
        if config.sticker.format == 'PNG':
            save_params['compress_level'] = 9 if optimize else 6
            
            # Try to reduce colors for smaller file size
            if optimize and image.mode == 'RGBA':
                # Check if we can use palette mode
                alpha = image.split()[3]
                if alpha.getextrema() == (255, 255):  # No transparency
                    # Convert to palette mode
                    image = image.convert('P', palette=Image.ADAPTIVE, colors=256)
                    save_params['format'] = 'PNG'
        
        elif config.sticker.format == 'WEBP':
            save_params['quality'] = config.sticker.quality
            save_params['method'] = 6
            save_params['lossless'] = False
        
        # Save image
        await asyncio.to_thread(
            image.save,
            output,
            **save_params
        )
        
        output.seek(0)
        
        # Check file size and optimize further if needed
        if optimize and len(output.getvalue()) > config.sticker.max_file_size_kb * 1024:
            output = await self._optimize_file_size(image)
        
        return output
    
    async def _optimize_file_size(self, image: Image.Image) -> io.BytesIO:
        """Aggressively optimize file size."""
        logger.debug("Optimizing file size...")
        
        output = io.BytesIO()
        
        # Try different optimization strategies
        strategies = [
            # Strategy 1: Reduce colors
            lambda img: img.convert('P', palette=Image.ADAPTIVE, colors=128),
            # Strategy 2: Reduce size
            lambda img: img.resize(
                (int(img.width * 0.9), int(img.height * 0.9)),
                Image.Resampling.LANCZOS
            ),
            # Strategy 3: More aggressive color reduction
            lambda img: img.convert('P', palette=Image.ADAPTIVE, colors=64),
        ]
        
        current_image = image
        
        for strategy in strategies:
            try:
                current_image = strategy(current_image)
                
                output = io.BytesIO()
                current_image.save(
                    output,
                    format='PNG',
                    optimize=True,
                    compress_level=9
                )
                output.seek(0)
                
                size_kb = len(output.getvalue()) / 1024
                if size_kb <= config.sticker.max_file_size_kb:
                    logger.debug(f"Optimized to {size_kb:.1f} KB")
                    break
                    
            except Exception as e:
                logger.warning(f"Optimization strategy failed: {e}")
        
        return output


# === Utility Functions ===

async def process_image_for_sticker(
    pil_image: Image.Image,
    background_style: BackgroundStyle = BackgroundStyle.TRANSPARENT,
    add_shadow: bool = False,
    add_outline: bool = False,
    remove_white_bg: bool = False,
    optimize_size: bool = True
) -> Optional[io.BytesIO]:
    """
    Process image for use as Telegram sticker.
    
    This is the main entry point that fixes the black screen issue.
    """
    processor = StickerProcessor()
    
    options = ProcessingOptions(
        background_style=background_style,
        add_shadow=add_shadow,
        add_outline=add_outline,
        remove_background=remove_white_bg or background_style == BackgroundStyle.TRANSPARENT,
        optimize_size=optimize_size
    )
    
    try:
        return await processor.process_image(pil_image, options)
    except Exception as e:
        logger.error(f"Failed to process sticker: {e}")
        return None


async def save_image_to_file(
    image_bytes: io.BytesIO,
    file_name: Optional[str] = None,
    storage_dir: str = "storage",
    create_subdirs: bool = True
) -> Optional[str]:
    """Save image bytes to file."""
    try:
        storage_path = Path(storage_dir)
        
        # Create subdirectory by date
        if create_subdirs:
            date_dir = datetime.now().strftime("%Y-%m-%d")
            storage_path = storage_path / date_dir
        
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if not file_name:
            image_bytes.seek(0)
            content_hash = hashlib.md5(image_bytes.read()).hexdigest()[:8]
            image_bytes.seek(0)
            file_name = f"sticker_{content_hash}_{uuid.uuid4().hex[:8]}.png"
        
        file_path = storage_path / file_name
        
        # Save file
        image_bytes.seek(0)
        await asyncio.to_thread(
            lambda: file_path.write_bytes(image_bytes.read())
        )
        
        if file_path.exists():
            logger.info(f"Image saved: {file_path}")
            return str(file_path)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return None


async def cleanup_old_files(storage_dir: str, days: int = 7) -> int:
    """Clean up old files from storage."""
    try:
        storage_path = Path(storage_dir)
        if not storage_path.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        # Find and delete old files
        for file_path in storage_path.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        # Remove empty directories
        for dir_path in sorted(storage_path.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception:
                    pass
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return 0


async def get_image_info(file_path: str) -> Optional[Dict[str, Any]]:
    """Get image file information."""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        image = await asyncio.to_thread(Image.open, str(path))
        
        # Get dominant colors
        color_thief = ColorThief(str(path))
        dominant_color = color_thief.get_color(quality=1)
        palette = color_thief.get_palette(color_count=5)
        
        return {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'size_bytes': path.stat().st_size,
            'size_kb': path.stat().st_size / 1024,
            'has_transparency': image.mode in ('RGBA', 'LA') or 
                               (image.mode == 'P' and 'transparency' in image.info),
            'dominant_color': dominant_color,
            'palette': palette
        }
        
    except Exception as e:
        logger.error(f"Failed to get image info: {e}")
        return None