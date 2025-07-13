import os
import time
import logging
import tempfile
import asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import librosa
import whisper
from langdetect import detect

logger = logging.getLogger(__name__)

class VoiceProcessor:
    """Service for processing voice audio and converting to text"""
    
    def __init__(self):
        self.model = None
        self.model_name = os.getenv("WHISPER_MODEL", "base")
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Audio processing settings
        self.target_sample_rate = 16000
        self.target_channels = 1
        
    async def warmup(self):
        """Warm up the service by loading the Whisper model"""
        logger.info("VoiceProcessor warming up...")
        try:
            await self._load_model()
            logger.info("VoiceProcessor warmed up successfully")
        except Exception as e:
            logger.error(f"VoiceProcessor warmup failed: {e}")
            # Continue without model for mock responses
    
    async def _load_model(self):
        """Load Whisper model in a thread"""
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            self.executor, 
            lambda: whisper.load_model(self.model_name)
        )
    
    def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio data for better recognition"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Load audio with pydub
                audio = AudioSegment.from_file(temp_file_path)
                
                # Convert to mono
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Normalize audio
                audio = normalize(audio)
                
                # Compress dynamic range
                audio = compress_dynamic_range(audio)
                
                # Convert to target sample rate
                audio = audio.set_frame_rate(self.target_sample_rate)
                
                # Convert to numpy array
                audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
                
                # Normalize to [-1, 1]
                if audio_array.dtype == np.int16:
                    audio_array = audio_array / 32768.0
                elif audio_array.dtype == np.int32:
                    audio_array = audio_array / 2147483648.0
                
                return audio_array
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Fallback: use librosa
            try:
                audio_array, _ = librosa.load(
                    temp_file_path, 
                    sr=self.target_sample_rate, 
                    mono=True
                )
                return audio_array
            except Exception as e2:
                logger.error(f"Fallback audio loading failed: {e2}")
                raise e
    
    def _enhance_audio_quality(self, audio_array: np.ndarray) -> np.ndarray:
        """Enhance audio quality for better recognition"""
        try:
            # Apply basic noise reduction
            # Simple spectral subtraction
            stft = librosa.stft(audio_array)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frames = int(0.5 * self.target_sample_rate / 512)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio_array
    
    def _transcribe_audio(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        if not self.model:
            # Mock response when model is not available
            return {
                "text": "This is a mock transcription. The actual audio processing is not available.",
                "confidence": 0.5,
                "language": "en"
            }
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_array,
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=False,
                verbose=False
            )
            
            # Extract text and confidence
            text = result["text"].strip()
            
            # Calculate average confidence from segments
            confidence = 0.0
            if "segments" in result and result["segments"]:
                confidences = [seg.get("avg_logprob", 0.0) for seg in result["segments"]]
                confidence = np.mean(confidences)
                # Convert log probability to confidence score
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))
            
            # Detect language
            detected_language = result.get("language", "en")
            
            return {
                "text": text,
                "confidence": confidence,
                "language": detected_language
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Try basic language detection on any available text
            try:
                detected_lang = detect(text) if 'text' in locals() else "en"
            except:
                detected_lang = "en"
            
            return {
                "text": "Transcription failed. Please try again.",
                "confidence": 0.0,
                "language": detected_lang
            }
    
    def _validate_audio(self, audio_data: bytes) -> bool:
        """Validate audio data"""
        if not audio_data:
            return False
        
        # Check minimum file size (at least 1KB)
        if len(audio_data) < 1024:
            return False
        
        # Check maximum file size (10MB)
        if len(audio_data) > 10 * 1024 * 1024:
            return False
        
        return True
    
    async def process_audio(self, audio_data: bytes, enhance_audio: bool = True) -> Dict[str, Any]:
        """Process audio and return transcribed text"""
        start_time = time.time()
        
        try:
            # Validate audio
            if not self._validate_audio(audio_data):
                raise ValueError("Invalid audio data")
            
            # Preprocess audio in thread pool
            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(
                self.executor,
                self._preprocess_audio,
                audio_data
            )
            
            # Enhance audio quality if requested
            if enhance_audio:
                audio_array = await loop.run_in_executor(
                    self.executor,
                    self._enhance_audio_quality,
                    audio_array
                )
            
            # Transcribe audio
            transcription_result = await loop.run_in_executor(
                self.executor,
                self._transcribe_audio,
                audio_array
            )
            
            processing_time = time.time() - start_time
            
            return {
                **transcription_result,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {
                "text": "Audio processing failed. Please try again.",
                "confidence": 0.0,
                "language": "en",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("VoiceProcessor cleaned up")