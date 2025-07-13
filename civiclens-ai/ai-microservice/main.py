from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Import services
from src.services.complaint_generator import ComplaintGenerator
from src.services.voice_processor import VoiceProcessor
from src.services.translator import TranslatorService
from src.services.ward_estimator import WardEstimator
from src.models.request_models import ComplaintRequest, TranslationRequest, VoiceRequest
from src.models.response_models import ComplaintResponse, TranslationResponse, VoiceResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
complaint_generator = None
voice_processor = None
translator_service = None
ward_estimator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    global complaint_generator, voice_processor, translator_service, ward_estimator
    
    try:
        # Initialize services
        logger.info("Initializing AI services...")
        
        complaint_generator = ComplaintGenerator()
        voice_processor = VoiceProcessor()
        translator_service = TranslatorService()
        ward_estimator = WardEstimator()
        
        # Warm up models
        logger.info("Warming up models...")
        await voice_processor.warmup()
        await complaint_generator.warmup()
        
        logger.info("AI services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize AI services: {e}")
        raise e
    finally:
        # Cleanup
        logger.info("Shutting down AI services...")
        if voice_processor:
            await voice_processor.cleanup()

# Create FastAPI app
app = FastAPI(
    title="CivicLens AI Microservice",
    description="AI-powered civic complaint processing service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CivicLens AI Microservice",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "complaint_generator": complaint_generator is not None,
            "voice_processor": voice_processor is not None,
            "translator": translator_service is not None,
            "ward_estimator": ward_estimator is not None
        }
    }

@app.post("/voice-to-text", response_model=VoiceResponse)
async def voice_to_text(audio: UploadFile = File(...)):
    """Convert voice audio to text"""
    try:
        if not voice_processor:
            raise HTTPException(status_code=503, detail="Voice processor not initialized")
        
        # Read audio file
        audio_data = await audio.read()
        
        # Process voice
        result = await voice_processor.process_audio(audio_data)
        
        return VoiceResponse(
            text=result["text"],
            confidence=result.get("confidence", 0.0),
            language=result.get("language", "en"),
            processing_time=result.get("processing_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

@app.post("/generate-complaint", response_model=ComplaintResponse)
async def generate_complaint(request: ComplaintRequest):
    """Generate formal complaint from description"""
    try:
        if not complaint_generator:
            raise HTTPException(status_code=503, detail="Complaint generator not initialized")
        
        # Generate complaint
        result = await complaint_generator.generate_complaint(
            description=request.description,
            address=request.address,
            category=request.category,
            language=request.language
        )
        
        # Get ward estimation
        ward_guess = None
        if ward_estimator:
            try:
                ward_guess = await ward_estimator.estimate_ward(request.address)
            except Exception as e:
                logger.warning(f"Ward estimation failed: {e}")
        
        # Translate if needed
        translated_text = None
        if translator_service and request.language != "en":
            try:
                translated_result = await translator_service.translate(
                    text=result["complaint_text"],
                    target_language=request.language
                )
                translated_text = translated_result["translated_text"]
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
        
        return ComplaintResponse(
            complaint_text=result["complaint_text"],
            translated_text=translated_text,
            summary=result.get("summary"),
            ward_guess=ward_guess,
            confidence=result.get("confidence", 0.0),
            processing_time=result.get("processing_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Complaint generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Complaint generation failed: {str(e)}")

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text to target language"""
    try:
        if not translator_service:
            raise HTTPException(status_code=503, detail="Translator service not initialized")
        
        result = await translator_service.translate(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language
        )
        
        return TranslationResponse(
            translated_text=result["translated_text"],
            source_language=result["source_language"],
            target_language=result["target_language"],
            confidence=result.get("confidence", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/estimate-ward")
async def estimate_ward(address: str):
    """Estimate ward from address"""
    try:
        if not ward_estimator:
            raise HTTPException(status_code=503, detail="Ward estimator not initialized")
        
        ward_guess = await ward_estimator.estimate_ward(address)
        
        return {
            "ward_guess": ward_guess,
            "address": address
        }
        
    except Exception as e:
        logger.error(f"Ward estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ward estimation failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )