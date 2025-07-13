from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class ComplaintCategory(str, Enum):
    """Complaint categories"""
    POTHOLE = "pothole"
    GARBAGE = "garbage"
    STREETLIGHT = "streetlight"
    WATER = "water"
    DRAINAGE = "drainage"
    OTHER = "other"

class Language(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"

class ComplaintRequest(BaseModel):
    """Request model for complaint generation"""
    description: str = Field(..., min_length=10, max_length=1000, description="Description of the civic issue")
    address: str = Field(..., min_length=5, max_length=500, description="Address where the issue is located")
    category: ComplaintCategory = Field(default=ComplaintCategory.OTHER, description="Category of the complaint")
    language: Language = Field(default=Language.ENGLISH, description="Target language for translation")

class TranslationRequest(BaseModel):
    """Request model for text translation"""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to translate")
    target_language: Language = Field(..., description="Target language for translation")
    source_language: Optional[Language] = Field(None, description="Source language (auto-detect if not provided)")

class VoiceRequest(BaseModel):
    """Request model for voice processing"""
    language: Optional[Language] = Field(Language.ENGLISH, description="Expected language of the audio")
    enhance_audio: bool = Field(True, description="Whether to enhance audio quality")