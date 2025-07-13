import os
import time
import logging
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

logger = logging.getLogger(__name__)

class ComplaintGenerator:
    """Service for generating formal complaints using OpenRouter API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = os.getenv("OPENROUTER_MODEL", "mistralai/mixtral-8x7b-instruct")
        self.client = httpx.AsyncClient(timeout=30.0)
        
        if not self.api_key:
            logger.warning("OpenRouter API key not provided. Using mock responses.")
    
    async def warmup(self):
        """Warm up the service"""
        logger.info("ComplaintGenerator warming up...")
        if self.api_key:
            try:
                # Test API connection
                await self._test_api_connection()
                logger.info("ComplaintGenerator warmed up successfully")
            except Exception as e:
                logger.error(f"ComplaintGenerator warmup failed: {e}")
        else:
            logger.info("ComplaintGenerator in mock mode")
    
    async def _test_api_connection(self):
        """Test API connection"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response = await self.client.get(
            f"{self.base_url}/models",
            headers=headers
        )
        response.raise_for_status()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API with retry logic"""
        if not self.api_key:
            return self._mock_response(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://civiclens.ai",
            "X-Title": "CivicLens AI",
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in drafting formal civic complaints. Generate clear, professional, and actionable complaints based on citizen descriptions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response when API key is not available"""
        return f"""
Dear Sir/Madam,

I am writing to formally report a civic issue that requires immediate attention.

Issue Description: Based on the citizen's report, there appears to be a significant civic infrastructure problem that needs to be addressed by the appropriate municipal authorities.

Location: As specified in the complaint details, this issue is located at the mentioned address and is affecting the daily lives of residents in the area.

Impact: This issue is causing inconvenience to the public and may pose safety concerns if not addressed promptly.

Request for Action: I respectfully request that the concerned department investigates this matter and takes appropriate remedial action at the earliest convenience.

Thank you for your attention to this matter.

Sincerely,
Concerned Citizen
        """.strip()
    
    def _create_complaint_prompt(self, description: str, address: str, category: str) -> str:
        """Create a prompt for generating formal complaints"""
        category_context = {
            "pothole": "road maintenance and infrastructure safety",
            "garbage": "waste management and public sanitation",
            "streetlight": "public lighting and street safety",
            "water": "water supply and plumbing infrastructure",
            "drainage": "drainage system and flood prevention",
            "other": "general civic infrastructure"
        }
        
        context = category_context.get(category, "civic infrastructure")
        
        prompt = f"""
Generate a formal civic complaint letter based on the following information:

Description: {description}
Address: {address}
Category: {category} ({context})

Requirements:
1. Use formal, professional language
2. Include proper salutation and closing
3. Clearly state the issue and its impact
4. Request specific action from authorities
5. Keep it concise but comprehensive
6. Make it actionable for government officials

Format as a complete formal letter suitable for submission to municipal authorities.
        """
        return prompt
    
    def _create_summary_prompt(self, complaint_text: str) -> str:
        """Create a prompt for generating complaint summary"""
        prompt = f"""
Create a brief, clear summary of the following civic complaint in 2-3 sentences:

{complaint_text}

Summary should:
- Highlight the main issue
- Mention the location
- Indicate urgency level
- Be easily understood by government officials
        """
        return prompt
    
    async def generate_complaint(
        self, 
        description: str, 
        address: str, 
        category: str = "other",
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate formal complaint from user description"""
        start_time = time.time()
        
        try:
            # Generate complaint prompt
            complaint_prompt = self._create_complaint_prompt(description, address, category)
            
            # Generate formal complaint
            complaint_text = await self._call_openrouter(complaint_prompt)
            
            # Generate summary
            summary_prompt = self._create_summary_prompt(complaint_text)
            summary = await self._call_openrouter(summary_prompt)
            
            processing_time = time.time() - start_time
            
            return {
                "complaint_text": complaint_text,
                "summary": summary,
                "confidence": 0.85,  # Mock confidence score
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Complaint generation failed: {e}")
            # Return fallback response
            return {
                "complaint_text": self._generate_fallback_complaint(description, address, category),
                "summary": f"Civic issue reported at {address}: {description[:100]}...",
                "confidence": 0.5,
                "processing_time": time.time() - start_time
            }
    
    def _generate_fallback_complaint(self, description: str, address: str, category: str) -> str:
        """Generate fallback complaint when API fails"""
        return f"""
Dear Sir/Madam,

I am writing to formally report a {category} issue that requires immediate attention.

Issue Description: {description}

Location: {address}

This issue is causing inconvenience to the public and residents in the area. I respectfully request that the concerned department investigates this matter and takes appropriate remedial action at the earliest convenience.

Thank you for your attention to this matter.

Sincerely,
Concerned Citizen
        """.strip()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()
        logger.info("ComplaintGenerator cleaned up")