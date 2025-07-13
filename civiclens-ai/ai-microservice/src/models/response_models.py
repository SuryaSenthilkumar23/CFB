from pydantic import BaseModel, Field
from typing import Optional
from .request_models import Language

class ComplaintResponse(BaseModel):
    """Response model for complaint generation"""
    complaint_text: str = Field(..., description="Formal complaint text")
    translated_text: Optional[str] = Field(None, description="Translated complaint text")
    summary: Optional[str] = Field(None, description="Summary of the complaint")
    ward_guess: Optional[str] = Field(None, description="Estimated ward information")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(0.0, ge=0.0, description="Processing time in seconds")

class TranslationResponse(BaseModel):
    """Response model for text translation"""
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Detected or provided source language")
    target_language: str = Field(..., description="Target language")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Translation confidence score")

class VoiceResponse(BaseModel):
    """Response model for voice processing"""
    text: str = Field(..., description="Transcribed text from audio")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Transcription confidence score")
    language: str = Field("en", description="Detected language")
    processing_time: float = Field(0.0, ge=0.0, description="Processing time in seconds")

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    code: Optional[str] = Field(None, description="Error code")