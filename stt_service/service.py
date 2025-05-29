"""
Advanced speech-to-text service using OpenAI Whisper.
Supports multiple languages, audio formats, and real-time processing.
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import shutil
import wave
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import deque
import warnings

import whisper
import torch
import torchaudio
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import detect_nonsilent
import librosa
import soundfile as sf

from logger import get_logger, log_execution_time, log_error
from config import config

logger = get_logger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class AudioValidationError(TranscriptionError):
    """Audio validation failed."""
    pass


class ModelLoadError(TranscriptionError):
    """Model loading failed."""
    pass


class AudioFormat(Enum):
    """Supported audio formats."""
    OGG = ".ogg"
    OGA = ".oga"
    MP3 = ".mp3"
    WAV = ".wav"
    M4A = ".m4a"
    FLAC = ".flac"
    WEBM = ".webm"
    OPUS = ".opus"


@dataclass
class AudioInfo:
    """Audio file information."""
    path: Path
    format: AudioFormat
    duration_sec: float
    sample_rate: int
    channels: int
    bitrate: Optional[int] = None
    size_mb: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """Check if audio meets requirements."""
        return (
            self.duration_sec > 0.1 and
            self.duration_sec <= config.security.max_audio_duration_sec and
            self.size_mb <= config.security.max_audio_size_mb
        )


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    text: str
    language: str
    language_probability: float
    duration: float
    segments: List[Dict[str, Any]] = field(default_factory=list)
    words: List[Dict[str, Any]] = field(default_factory=list)
    process_time: float = 0.0
    audio_info: Optional[AudioInfo] = None
    
    @property
    def confidence(self) -> float:
        """Calculate average confidence from segments."""
        if not self.segments:
            return self.language_probability
        
        confidences = [seg.get('no_speech_prob', 0) for seg in self.segments]
        return 1.0 - (sum(confidences) / len(confidences)) if confidences else 0.0
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            'text': self.text,
            'language': self.language,
            'language_probability': self.language_probability,
            'duration': self.duration,
            'confidence': self.confidence,
            'process_time': self.process_time,
            'segments_count': len(self.segments),
            'words_count': len(self.words)
        })


class AudioProcessor:
    """Advanced audio processing utilities."""
    
    def __init__(self):
        """Initialize processor."""
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def validate_audio(self, file_path: Union[str, Path]) -> AudioInfo:
        """Validate and get audio information."""
        path = Path(file_path)
        
        # Check file exists
        if not path.exists():
            raise AudioValidationError(f"File not found: {file_path}")
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > config.security.max_audio_size_mb:
            raise AudioValidationError(
                f"File too large: {size_mb:.1f} MB "
                f"(max: {config.security.max_audio_size_mb} MB)"
            )
        
        # Check format
        file_ext = path.suffix.lower()
        try:
            audio_format = AudioFormat(file_ext)
        except ValueError:
            raise AudioValidationError(
                f"Unsupported format: {file_ext}. "
                f"Supported: {', '.join([f.value for f in AudioFormat])}"
            )
        
        # Get audio info
        try:
            audio_info = await self._get_audio_info(path, audio_format, size_mb)
            
            # Validate duration
            if audio_info.duration_sec > config.security.max_audio_duration_sec:
                raise AudioValidationError(
                    f"Audio too long: {audio_info.duration_sec:.1f} sec "
                    f"(max: {config.security.max_audio_duration_sec} sec)"
                )
            
            if audio_info.duration_sec < 0.1:
                raise AudioValidationError("Audio too short (< 0.1 sec)")
            
            return audio_info
            
        except Exception as e:
            raise AudioValidationError(f"Failed to process audio: {e}")
    
    async def _get_audio_info(
        self,
        path: Path,
        audio_format: AudioFormat,
        size_mb: float
    ) -> AudioInfo:
        """Extract audio information."""
        # Try multiple methods for robustness
        
        # Method 1: Try with librosa (most reliable for various formats)
        try:
            return await self._get_info_librosa(path, audio_format, size_mb)
        except Exception as e:
            logger.debug(f"Librosa failed: {e}")
        
        # Method 2: Try with pydub
        try:
            return await self._get_info_pydub(path, audio_format, size_mb)
        except Exception as e:
            logger.debug(f"Pydub failed: {e}")
        
        # Method 3: Try with torchaudio
        try:
            return await self._get_info_torchaudio(path, audio_format, size_mb)
        except Exception as e:
            logger.debug(f"Torchaudio failed: {e}")
        
        raise AudioValidationError("Failed to read audio with any method")
    
    async def _get_info_librosa(
        self,
        path: Path,
        audio_format: AudioFormat,
        size_mb: float
    ) -> AudioInfo:
        """Get audio info using librosa."""
        def _load():
            y, sr = librosa.load(str(path), sr=None, mono=False)
            duration = librosa.get_duration(y=y, sr=sr)
            channels = 1 if y.ndim == 1 else y.shape[0]
            return duration, sr, channels
        
        duration, sample_rate, channels = await asyncio.to_thread(_load)
        
        return AudioInfo(
            path=path,
            format=audio_format,
            duration_sec=duration,
            sample_rate=sample_rate,
            channels=channels,
            size_mb=size_mb
        )
    
    async def _get_info_pydub(
        self,
        path: Path,
        audio_format: AudioFormat,
        size_mb: float
    ) -> AudioInfo:
        """Get audio info using pydub."""
        audio = await asyncio.to_thread(AudioSegment.from_file, str(path))
        
        return AudioInfo(
            path=path,
            format=audio_format,
            duration_sec=len(audio) / 1000.0,
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            bitrate=audio.frame_rate * audio.sample_width * 8 * audio.channels,
            size_mb=size_mb
        )
    
    async def _get_info_torchaudio(
        self,
        path: Path,
        audio_format: AudioFormat,
        size_mb: float
    ) -> AudioInfo:
        """Get audio info using torchaudio."""
        info = await asyncio.to_thread(torchaudio.info, str(path))
        
        return AudioInfo(
            path=path,
            format=audio_format,
            duration_sec=info.num_frames / info.sample_rate,
            sample_rate=info.sample_rate,
            channels=info.num_channels,
            bitrate=info.bits_per_sample * info.sample_rate * info.num_channels,
            size_mb=size_mb
        )
    
    async def prepare_for_whisper(
        self,
        file_path: Union[str, Path],
        target_sample_rate: int = 16000,
        optimize: bool = True
    ) -> str:
        """Prepare audio file for Whisper processing."""
        path = Path(file_path)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            dir=path.parent
        ) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            audio = await asyncio.to_thread(AudioSegment.from_file, str(path))
            
            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.debug("Converted to mono")
            
            # Resample if needed
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
                logger.debug(f"Resampled to {target_sample_rate} Hz")
            
            if optimize:
                # Normalize volume
                audio = await self._normalize_audio(audio)
                
                # Remove silence
                audio = await self._trim_silence(audio)
                
                # Apply noise reduction if very quiet
                if audio.dBFS < -30:
                    audio = await self._reduce_noise(audio)
            
            # Export
            await asyncio.to_thread(
                audio.export,
                tmp_path,
                format='wav',
                parameters=['-q:a', '0']
            )
            
            logger.debug(f"Audio prepared: {tmp_path}")
            return tmp_path
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise TranscriptionError(f"Failed to prepare audio: {e}")
    
    async def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio volume."""
        # Calculate how much to adjust
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        
        # Limit adjustment to prevent distortion
        change_in_dBFS = max(-10, min(10, change_in_dBFS))
        
        if abs(change_in_dBFS) > 0.5:
            return audio + change_in_dBFS
        return audio
    
    async def _trim_silence(
        self,
        audio: AudioSegment,
        silence_thresh: int = -40,
        chunk_size: int = 10
    ) -> AudioSegment:
        """Remove silence from beginning and end."""
        nonsilent_chunks = detect_nonsilent(
            audio,
            min_silence_len=100,
            silence_thresh=silence_thresh,
            seek_step=chunk_size
        )
        
        if nonsilent_chunks:
            start = nonsilent_chunks[0][0]
            end = nonsilent_chunks[-1][1]
            return audio[start:end]
        
        return audio
    
    async def _reduce_noise(self, audio: AudioSegment) -> AudioSegment:
        """Simple noise reduction."""
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Apply simple high-pass filter to reduce low-frequency noise
        # This is a very basic implementation
        from scipy import signal
        
        # Design a high-pass filter
        nyquist = audio.frame_rate / 2
        cutoff = 80 / nyquist  # 80 Hz cutoff
        b, a = signal.butter(5, cutoff, btype='high')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, samples)
        
        # Convert back to AudioSegment
        filtered_audio = audio._spawn(filtered.astype(np.int16).tobytes())
        
        return filtered_audio
    
    async def split_audio(
        self,
        file_path: Union[str, Path],
        chunk_duration_sec: float = 30.0
    ) -> List[str]:
        """Split audio into chunks for processing."""
        path = Path(file_path)
        audio = await asyncio.to_thread(AudioSegment.from_file, str(path))
        
        chunks = []
        chunk_length_ms = int(chunk_duration_sec * 1000)
        
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            
            # Save chunk
            chunk_path = path.parent / f"{path.stem}_chunk_{i // chunk_length_ms}.wav"
            await asyncio.to_thread(
                chunk.export,
                str(chunk_path),
                format='wav'
            )
            chunks.append(str(chunk_path))
        
        return chunks


class ModelManager:
    """Manages Whisper models."""
    
    def __init__(self):
        """Initialize model manager."""
        self.models: Dict[str, whisper.Whisper] = {}
        self._lock = asyncio.Lock()
        self.device = None
        self.compute_type = None
    
    async def load_model(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        download_root: Optional[str] = None
    ) -> whisper.Whisper:
        """Load or get cached Whisper model."""
        model_size = model_size or config.whisper.get_model_name()
        device = device or config.whisper.device.value
        download_root = download_root or str(config.paths.models_dir / "whisper")
        
        cache_key = f"{model_size}_{device}"
        
        async with self._lock:
            if cache_key in self.models:
                return self.models[cache_key]
            
            logger.info(f"Loading Whisper model '{model_size}' on '{device}'")
            
            try:
                # Load model in thread
                model = await asyncio.to_thread(
                    whisper.load_model,
                    model_size,
                    device=device,
                    download_root=download_root,
                    in_memory=True
                )
                
                # Optimize model
                model = await self._optimize_model(model, device)
                
                # Cache model
                self.models[cache_key] = model
                self.device = device
                
                # Warmup
                await self._warmup_model(model)
                
                logger.info(f"Whisper model '{model_size}' loaded successfully")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise ModelLoadError(f"Failed to load model: {e}")
    
    async def _optimize_model(self, model: whisper.Whisper, device: str) -> whisper.Whisper:
        """Apply optimizations to model."""
        if device == 'cuda' and torch.cuda.is_available():
            # Enable mixed precision for faster inference
            model = model.half()
            
            # Compile model with torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    logger.debug("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}")
        
        return model
    
    async def _warmup_model(self, model: whisper.Whisper):
        """Warmup model with dummy audio."""
        try:
            logger.debug("Warming up Whisper model...")
            
            # Create 1 second of silence
            dummy_audio = np.zeros(16000, dtype=np.float32)
            
            # Run transcription
            await asyncio.to_thread(
                model.transcribe,
                dummy_audio,
                language='en',
                fp16=self.device == 'cuda'
            )
            
            logger.debug("Model warmed up")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def clear_cache(self):
        """Clear model cache."""
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LanguageDetector:
    """Language detection utilities."""
    
    # Common language codes and names
    LANGUAGES = {
        'en': 'English',
        'zh': 'Chinese',
        'es': 'Spanish',
        'hi': 'Hindi',
        'ar': 'Arabic',
        'bn': 'Bengali',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'pa': 'Punjabi',
        'de': 'German',
        'ko': 'Korean',
        'fr': 'French',
        'it': 'Italian',
        'tr': 'Turkish',
        'pl': 'Polish',
        'uk': 'Ukrainian',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'fi': 'Finnish',
        'he': 'Hebrew',
        'el': 'Greek',
        'ro': 'Romanian',
        'hu': 'Hungarian',
        'cs': 'Czech',
        'da': 'Danish',
        'no': 'Norwegian',
        'th': 'Thai',
        'id': 'Indonesian',
        'vi': 'Vietnamese'
    }
    
    @classmethod
    def get_language_name(cls, code: str) -> str:
        """Get language name from code."""
        return cls.LANGUAGES.get(code, code.upper())
    
    @classmethod
    async def detect_language(
        cls,
        model: whisper.Whisper,
        audio_path: str
    ) -> Dict[str, float]:
        """Detect language probabilities."""
        # Load audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Detect language
        _, probs = await asyncio.to_thread(
            model.detect_language,
            mel
        )
        
        # Sort by probability
        language_probs = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top 5 with names
        return {
            code: {
                'probability': prob,
                'name': cls.get_language_name(code)
            }
            for code, prob in language_probs[:5]
        }


class TranscriptionService:
    """Main transcription service."""
    
    def __init__(self):
        """Initialize service."""
        self.model_manager = ModelManager()
        self.audio_processor = AudioProcessor()
        self.language_detector = LanguageDetector()
        self._stats = {
            'total_transcriptions': 0,
            'total_duration': 0.0,
            'total_process_time': 0.0,
            'languages': {}
        }
    
    @log_execution_time
    @log_error("transcription")
    async def transcribe(
        self,
        audio_file_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = 'transcribe',
        initial_prompt: Optional[str] = None,
        temperature: Union[float, List[float]] = 0.0,
        verbose: bool = False,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (None for auto-detect)
            task: 'transcribe' or 'translate'
            initial_prompt: Initial prompt for better context
            temperature: Temperature for sampling
            verbose: Whether to print progress
            **kwargs: Additional Whisper parameters
            
        Returns:
            Transcription result
        """
        start_time = time.time()
        temp_file = None
        
        try:
            # Validate audio
            audio_info = await self.audio_processor.validate_audio(audio_file_path)
            logger.info(
                f"Transcribing audio: {audio_info.duration_sec:.1f} sec, "
                f"{audio_info.sample_rate} Hz, {audio_info.format.value}"
            )
            
            # Load model
            model = await self.model_manager.load_model()
            
            # Prepare audio if needed
            if (audio_info.format in [AudioFormat.OGG, AudioFormat.OPUS] or
                audio_info.sample_rate != 16000 or
                audio_info.channels > 1):
                logger.debug("Preparing audio for Whisper...")
                temp_file = await self.audio_processor.prepare_for_whisper(
                    audio_file_path,
                    optimize=True
                )
                process_file = temp_file
            else:
                process_file = str(audio_file_path)
            
            # Auto-detect language if not specified
            if not language:
                logger.debug("Detecting language...")
                lang_probs = await self.language_detector.detect_language(
                    model, process_file
                )
                language = list(lang_probs.keys())[0]
                language_prob = lang_probs[language]['probability']
                logger.info(f"Detected language: {language} ({language_prob:.2%})")
            else:
                language_prob = 1.0
            
            # Set initial prompt based on language
            if not initial_prompt and language == 'ru':
                initial_prompt = "Это голосовое сообщение для создания стикера."
            
            # Transcribe
            logger.debug("Starting transcription...")
            result = await self._transcribe_with_options(
                model=model,
                audio_path=process_file,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                temperature=temperature,
                verbose=verbose,
                **kwargs
            )
            
            # Process result
            transcript = result.get('text', '').strip()
            
            if not transcript:
                logger.warning("Empty transcription result")
                raise TranscriptionError("No speech detected")
            
            # Clean transcript
            transcript = await self._clean_transcript(transcript, language)
            
            # Create result
            process_time = time.time() - start_time
            
            transcription_result = TranscriptionResult(
                text=transcript,
                language=language,
                language_probability=language_prob,
                duration=audio_info.duration_sec,
                segments=result.get('segments', []),
                process_time=process_time,
                audio_info=audio_info
            )
            
            # Extract word timestamps if available
            if config.whisper.word_timestamps:
                transcription_result.words = await self._extract_words(result)
            
            # Update statistics
            self._update_stats(transcription_result)
            
            logger.info(
                f"Transcription completed in {process_time:.2f}s. "
                f"Text length: {len(transcript)} chars, "
                f"Confidence: {transcription_result.confidence:.2%}"
            )
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Transcription failed: {e}")
            
        finally:
            # Cleanup
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
    
    async def _transcribe_with_options(
        self,
        model: whisper.Whisper,
        audio_path: str,
        **options
    ) -> Dict[str, Any]:
        """Transcribe with optimized options."""
        # Set device-specific options
        device = self.model_manager.device or 'cpu'
        
        # Default options
        default_options = {
            'fp16': device in ['cuda', 'mps'] and model.device != 'cpu',
            'verbose': False,
            'condition_on_previous_text': True,
            'no_speech_threshold': 0.6,
            'logprob_threshold': -1.0,
            'compression_ratio_threshold': 2.4,
            'beam_size': 5,
            'best_of': 5,
            'patience': 1.0,
            'length_penalty': 1.0,
            'suppress_tokens': "-1",
            'suppress_blank': True,
            'without_timestamps': False,
            'max_initial_timestamp': 1.0,
            'word_timestamps': False
        }
        
        # Merge with provided options
        options = {**default_options, **options}
        
        # Run transcription
        result = await asyncio.to_thread(
            model.transcribe,
            audio_path,
            **options
        )
        
        return result
    
    async def _clean_transcript(self, text: str, language: str) -> str:
        """Clean and improve transcript."""
        # Remove common artifacts
        artifacts = [
            '[BLANK_AUDIO]',
            '[MUSIC]',
            '[APPLAUSE]',
            '(unintelligible)',
            '(inaudible)',
            '...',
            '♪',
            '♫'
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Language-specific cleaning
        if language == 'ru':
            # Fix common Russian transcription issues
            replacements = {
                ' ,': ',',
                ' .': '.',
                ' !': '!',
                ' ?': '?',
                '  ': ' '
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
        
        # Remove repeated punctuation
        import re
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    async def _extract_words(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from segments."""
        words = []
        
        for segment in result.get('segments', []):
            segment_words = segment.get('words', [])
            for word in segment_words:
                words.append({
                    'word': word.get('word', ''),
                    'start': word.get('start', 0),
                    'end': word.get('end', 0),
                    'probability': word.get('probability', 0)
                })
        
        return words
    
    def _update_stats(self, result: TranscriptionResult):
        """Update service statistics."""
        self._stats['total_transcriptions'] += 1
        self._stats['total_duration'] += result.duration
        self._stats['total_process_time'] += result.process_time
        
        # Language statistics
        if result.language not in self._stats['languages']:
            self._stats['languages'][result.language] = 0
        self._stats['languages'][result.language] += 1
    
    async def transcribe_long_audio(
        self,
        audio_file_path: Union[str, Path],
        chunk_duration: float = 30.0,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe long audio by splitting into chunks."""
        # Split audio
        chunks = await self.audio_processor.split_audio(
            audio_file_path,
            chunk_duration
        )
        
        logger.info(f"Split audio into {len(chunks)} chunks")
        
        # Transcribe each chunk
        all_segments = []
        all_text = []
        total_duration = 0
        
        for i, chunk_path in enumerate(chunks):
            logger.debug(f"Transcribing chunk {i+1}/{len(chunks)}")
            
            try:
                result = await self.transcribe(chunk_path, **kwargs)
                all_text.append(result.text)
                all_segments.extend(result.segments)
                total_duration += result.duration
            finally:
                # Clean up chunk
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
        
        # Combine results
        combined_text = ' '.join(all_text)
        
        return TranscriptionResult(
            text=combined_text,
            language=kwargs.get('language', 'auto'),
            language_probability=1.0,
            duration=total_duration,
            segments=all_segments,
            process_time=time.time()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = self._stats.copy()
        
        if stats['total_transcriptions'] > 0:
            stats['avg_duration'] = stats['total_duration'] / stats['total_transcriptions']
            stats['avg_process_time'] = stats['total_process_time'] / stats['total_transcriptions']
            stats['real_time_factor'] = stats['total_process_time'] / stats['total_duration']
        
        return stats


# === Global instance ===
_service: Optional[TranscriptionService] = None


async def get_service() -> TranscriptionService:
    """Get or create global service instance."""
    global _service
    if _service is None:
        _service = TranscriptionService()
    return _service


# === Public API ===

async def transcribe_audio(
    audio_file_path: Union[str, Path],
    language: Optional[str] = None,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Transcribe audio file.
    
    Args:
        audio_file_path: Path to audio file
        language: Language code or None for auto-detect
        **kwargs: Additional parameters
        
    Returns:
        Transcription result dict or None on error
    """
    try:
        service = await get_service()
        result = await service.transcribe(
            audio_file_path,
            language=language,
            **kwargs
        )
        
        return {
            'text': result.text,
            'language': result.language,
            'language_probability': result.language_probability,
            'duration': result.duration,
            'process_time': result.process_time,
            'segments': result.segments,
            'confidence': result.confidence
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None


async def get_audio_languages(audio_file_path: Union[str, Path]) -> Dict[str, float]:
    """
    Detect languages in audio file.
    
    Returns:
        Dict of language codes to probabilities
    """
    try:
        service = await get_service()
        model = await service.model_manager.load_model()
        
        # Prepare audio
        temp_file = await service.audio_processor.prepare_for_whisper(audio_file_path)
        
        try:
            lang_probs = await service.language_detector.detect_language(model, temp_file)
            return {
                code: info['probability']
                for code, info in lang_probs.items()
            }
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return {}


async def load_whisper_model():
    """Pre-load Whisper model."""
    service = await get_service()
    await service.model_manager.load_model()
    logger.info("Whisper model loaded")


async def get_transcription_stats() -> Dict[str, Any]:
    """Get transcription statistics."""
    service = await get_service()
    return service.get_stats()