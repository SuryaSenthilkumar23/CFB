import os
import time
import logging
from typing import Dict, Any, Optional
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectError
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TranslatorService:
    """Service for translating text between languages"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Language mappings
        self.language_mapping = {
            "en": "english",
            "hi": "hindi", 
            "ta": "tamil",
            "te": "telugu",
            "bn": "bengali",
            "mr": "marathi",
            "gu": "gujarati",
            "kn": "kannada",
            "ml": "malayalam",
            "pa": "punjabi",
            "or": "odia",
            "as": "assamese",
            "ur": "urdu"
        }
        
        # Supported languages for translation
        self.supported_languages = ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"]
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text"""
        try:
            if not text or not text.strip():
                return "en"
            
            detected = detect(text)
            # Map detected language to our supported languages
            if detected in self.supported_languages:
                return detected
            
            # Default to English if detection fails or language not supported
            return "en"
            
        except LangDetectError as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"
    
    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate text using Google Translator"""
        try:
            if not text or not text.strip():
                return {
                    "translated_text": text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "confidence": 0.0
                }
            
            # If source and target are the same, return original text
            if source_lang == target_lang:
                return {
                    "translated_text": text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "confidence": 1.0
                }
            
            # Create translator
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            
            # Translate text
            translated = translator.translate(text)
            
            # Calculate confidence (mock implementation)
            confidence = 0.85 if len(translated) > 0 else 0.0
            
            return {
                "translated_text": translated,
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "translated_text": text,  # Return original text on failure
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _chunk_text(self, text: str, max_length: int = 4000) -> list:
        """Split text into chunks for translation"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _translate_chunks(self, chunks: list, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate text chunks and combine results"""
        try:
            translated_chunks = []
            total_confidence = 0.0
            
            for chunk in chunks:
                result = self._translate_text(chunk, source_lang, target_lang)
                translated_chunks.append(result["translated_text"])
                total_confidence += result["confidence"]
            
            # Combine translated chunks
            combined_text = " ".join(translated_chunks)
            average_confidence = total_confidence / len(chunks) if chunks else 0.0
            
            return {
                "translated_text": combined_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence": average_confidence
            }
            
        except Exception as e:
            logger.error(f"Chunk translation failed: {e}")
            return {
                "translated_text": " ".join(chunks),  # Return original chunks
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Translate text to target language"""
        start_time = time.time()
        
        try:
            # Validate input
            if not text or not text.strip():
                return {
                    "translated_text": text,
                    "source_language": source_language or "en",
                    "target_language": target_language,
                    "confidence": 0.0
                }
            
            # Detect source language if not provided
            if not source_language:
                source_language = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._detect_language,
                    text
                )
            
            # Validate languages
            if target_language not in self.supported_languages:
                logger.warning(f"Unsupported target language: {target_language}")
                target_language = "en"
            
            if source_language not in self.supported_languages:
                logger.warning(f"Unsupported source language: {source_language}")
                source_language = "en"
            
            # Split text into chunks if too long
            chunks = self._chunk_text(text)
            
            # Translate chunks
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._translate_chunks,
                chunks,
                source_language,
                target_language
            )
            
            # Add processing time
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Translation service failed: {e}")
            return {
                "translated_text": text,  # Return original text on failure
                "source_language": source_language or "en",
                "target_language": target_language,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        return self.language_mapping
    
    async def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._detect_language,
                text
            )
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("TranslatorService cleaned up")