import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
import googlemaps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

logger = logging.getLogger(__name__)

class WardEstimator:
    """Service for estimating ward information from addresses"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.gmaps = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize Google Maps client if API key is available
        if self.api_key:
            try:
                self.gmaps = googlemaps.Client(key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Google Maps client: {e}")
                self.gmaps = None
        else:
            logger.warning("Google Maps API key not provided. Using mock responses.")
        
        # Ward mapping patterns for major Indian cities
        self.ward_patterns = {
            "mumbai": {
                "andheri": "Ward 2, Zone 4",
                "bandra": "Ward 4, Zone 3",
                "borivali": "Ward 3, Zone 5",
                "malad": "Ward 5, Zone 5",
                "churchgate": "Ward 1, Zone 1",
                "colaba": "Ward 1, Zone 1",
                "dadar": "Ward 2, Zone 2",
                "kurla": "Ward 3, Zone 3",
                "powai": "Ward 4, Zone 4",
                "versova": "Ward 2, Zone 4"
            },
            "delhi": {
                "cp": "Ward 1, Zone 1",
                "connaught place": "Ward 1, Zone 1",
                "karol bagh": "Ward 5, Zone 2",
                "lajpat nagar": "Ward 8, Zone 3",
                "rohini": "Ward 12, Zone 4",
                "dwarka": "Ward 15, Zone 5",
                "vasant kunj": "Ward 10, Zone 4",
                "saket": "Ward 9, Zone 3",
                "greater kailash": "Ward 7, Zone 3",
                "defence colony": "Ward 6, Zone 2"
            },
            "bangalore": {
                "koramangala": "Ward 85, Zone 5",
                "whitefield": "Ward 120, Zone 8",
                "indiranagar": "Ward 86, Zone 5",
                "jayanagar": "Ward 69, Zone 4",
                "malleshwaram": "Ward 10, Zone 1",
                "rajajinagar": "Ward 11, Zone 1",
                "hebbal": "Ward 7, Zone 1",
                "electronic city": "Ward 195, Zone 11",
                "mg road": "Ward 81, Zone 5",
                "brigade road": "Ward 82, Zone 5"
            },
            "hyderabad": {
                "hitech city": "Ward 45, Zone 5",
                "gachibowli": "Ward 46, Zone 5",
                "jubilee hills": "Ward 40, Zone 4",
                "banjara hills": "Ward 38, Zone 4",
                "secunderabad": "Ward 25, Zone 3",
                "kukatpally": "Ward 55, Zone 6",
                "madhapur": "Ward 44, Zone 5",
                "begumpet": "Ward 35, Zone 4",
                "abids": "Ward 30, Zone 3",
                "charminar": "Ward 20, Zone 2"
            },
            "chennai": {
                "anna nagar": "Ward 101, Zone 6",
                "t nagar": "Ward 127, Zone 9",
                "adyar": "Ward 174, Zone 13",
                "velachery": "Ward 183, Zone 14",
                "tambaram": "Ward 190, Zone 15",
                "porur": "Ward 108, Zone 7",
                "omr": "Ward 185, Zone 14",
                "egmore": "Ward 95, Zone 5",
                "mylapore": "Ward 152, Zone 11",
                "guindy": "Ward 158, Zone 12"
            },
            "kolkata": {
                "park street": "Ward 65, Zone 5",
                "salt lake": "Ward 108, Zone 8",
                "ballygunge": "Ward 75, Zone 6",
                "alipore": "Ward 82, Zone 6",
                "howrah": "Ward 25, Zone 2",
                "rajarhat": "Ward 115, Zone 9",
                "new town": "Ward 120, Zone 9",
                "gariahat": "Ward 78, Zone 6",
                "jadavpur": "Ward 85, Zone 7",
                "dum dum": "Ward 35, Zone 3"
            }
        }
    
    def _extract_city_from_address(self, address: str) -> Optional[str]:
        """Extract city name from address"""
        address_lower = address.lower()
        
        # Check for city names in address
        cities = ["mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata"]
        for city in cities:
            if city in address_lower:
                return city
        
        # Check for common city aliases
        aliases = {
            "bombay": "mumbai",
            "new delhi": "delhi",
            "bengaluru": "bangalore",
            "madras": "chennai",
            "calcutta": "kolkata"
        }
        
        for alias, city in aliases.items():
            if alias in address_lower:
                return city
        
        return None
    
    def _extract_locality_from_address(self, address: str) -> Optional[str]:
        """Extract locality/area name from address"""
        address_lower = address.lower()
        
        # Common patterns to identify localities
        patterns = [
            r'\b([a-zA-Z\s]+)\s+(?:road|street|avenue|lane|colony|nagar|puram|ganj|park|plaza|sector|block)\b',
            r'\b([a-zA-Z\s]+)\s+(?:extension|ext|phase|stage)\b',
            r'\bnear\s+([a-zA-Z\s]+)\b',
            r'\b([a-zA-Z\s]+)\s+(?:metro|station|junction|circle)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, address_lower)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _estimate_ward_from_patterns(self, address: str) -> Optional[str]:
        """Estimate ward using pattern matching"""
        city = self._extract_city_from_address(address)
        if not city or city not in self.ward_patterns:
            return None
        
        locality = self._extract_locality_from_address(address)
        if not locality:
            return None
        
        # Check if locality matches any ward pattern
        city_patterns = self.ward_patterns[city]
        for pattern, ward in city_patterns.items():
            if pattern in locality.lower():
                return ward
        
        return None
    
    def _geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode address to get coordinates"""
        if not self.gmaps:
            return None
        
        try:
            geocode_result = self.gmaps.geocode(address)
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                return (location['lat'], location['lng'])
        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
        
        return None
    
    def _reverse_geocode(self, lat: float, lng: float) -> Optional[Dict[str, Any]]:
        """Reverse geocode coordinates to get address components"""
        if not self.gmaps:
            return None
        
        try:
            reverse_geocode_result = self.gmaps.reverse_geocode((lat, lng))
            if reverse_geocode_result:
                return reverse_geocode_result[0]
        except Exception as e:
            logger.error(f"Reverse geocoding failed: {e}")
        
        return None
    
    def _extract_administrative_info(self, geocode_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract administrative information from geocode result"""
        components = geocode_result.get('address_components', [])
        
        info = {
            'locality': '',
            'sublocality': '',
            'administrative_area_level_1': '',
            'administrative_area_level_2': '',
            'administrative_area_level_3': '',
            'country': ''
        }
        
        for component in components:
            types = component.get('types', [])
            long_name = component.get('long_name', '')
            
            for type_name in types:
                if type_name in info:
                    info[type_name] = long_name
                    break
        
        return info
    
    def _generate_ward_estimate(self, address: str, admin_info: Dict[str, str]) -> str:
        """Generate ward estimate based on administrative information"""
        # Try pattern matching first
        pattern_ward = self._estimate_ward_from_patterns(address)
        if pattern_ward:
            return pattern_ward
        
        # Generate estimate based on administrative info
        locality = admin_info.get('locality', '')
        sublocality = admin_info.get('sublocality', '')
        area_level_2 = admin_info.get('administrative_area_level_2', '')
        area_level_3 = admin_info.get('administrative_area_level_3', '')
        
        # Create a generic ward estimate
        ward_info = []
        if sublocality:
            ward_info.append(f"Sublocality: {sublocality}")
        if locality:
            ward_info.append(f"Locality: {locality}")
        if area_level_3:
            ward_info.append(f"Area: {area_level_3}")
        if area_level_2:
            ward_info.append(f"District: {area_level_2}")
        
        if ward_info:
            return " | ".join(ward_info)
        
        return "Ward information not available"
    
    async def estimate_ward(self, address: str) -> str:
        """Estimate ward information from address"""
        if not address or not address.strip():
            return "Invalid address"
        
        try:
            # Run geocoding in thread pool
            loop = asyncio.get_event_loop()
            
            # Get coordinates
            coordinates = await loop.run_in_executor(
                self.executor,
                self._geocode_address,
                address
            )
            
            if coordinates:
                lat, lng = coordinates
                
                # Get detailed address components
                geocode_result = await loop.run_in_executor(
                    self.executor,
                    self._reverse_geocode,
                    lat,
                    lng
                )
                
                if geocode_result:
                    admin_info = self._extract_administrative_info(geocode_result)
                    
                    # Generate ward estimate
                    ward_estimate = self._generate_ward_estimate(address, admin_info)
                    return ward_estimate
            
            # Fallback to pattern matching
            pattern_ward = self._estimate_ward_from_patterns(address)
            if pattern_ward:
                return pattern_ward
            
            # Generate basic estimate from address
            city = self._extract_city_from_address(address)
            if city:
                return f"City: {city.title()} | Ward information requires more specific address"
            
            return "Ward information not available for this address"
            
        except Exception as e:
            logger.error(f"Ward estimation failed: {e}")
            return "Ward estimation failed"
    
    async def get_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for address"""
        if not address or not address.strip():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._geocode_address,
                address
            )
        except Exception as e:
            logger.error(f"Coordinate lookup failed: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("WardEstimator cleaned up")