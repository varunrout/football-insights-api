"""
Environment configuration management for Football Insights API.
Handles loading and validating environment variables for different deployment scenarios.
"""

import os
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings as PydanticBaseSettings, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Environment(str, Enum):
    """Available deployment environments"""
    TEST = "TEST"   # For testing with mock or limited data
    DEV = "DEV"     # Development with real data
    QUAL = "QUAL"   # Pre-production validation (mirror of PROD)
    PROD = "PROD"   # Production deployment

class Settings(PydanticBaseSettings):
    """Application settings loaded from environment variables"""
    
    # Environment settings
    ENVIRONMENT: Environment = Field(Environment.DEV, description="Deployment environment")
    
    # Data settings
    DATA_CACHE_DIR: str = Field("data_cache", description="Directory for cached data")
    USE_MOCK_DATA: bool = Field(False, description="Use mock data instead of real data")
    MAX_COMPETITIONS: int = Field(10, description="Maximum number of competitions to process")
    MAX_MATCHES_PER_COMPETITION: int = Field(5, description="Maximum matches per competition")
    
    # API settings
    API_HOST: str = Field("0.0.0.0", description="API host address")
    API_PORT: int = Field(8000, description="API port")
    ENABLE_CORS: bool = Field(True, description="Enable CORS for API")
    ALLOWED_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://127.0.0.1:3000"], 
        description="Allowed origins for CORS"
    )
    
    # Logging
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_TO_FILE: bool = Field(False, description="Log to file")
    LOG_FILE: str = Field("logs/api.log", description="Log file path")
    
    # Performance
    CACHE_EXPIRY_SECONDS: int = Field(3600, description="Cache expiry in seconds")
    USE_COMPRESSION: bool = Field(True, description="Use compression for API responses")
    
    @validator("DATA_CACHE_DIR")
    def validate_data_cache_dir(cls, v):
        """Ensure data cache directory exists"""
        path = Path(v)
        path.mkdir(exist_ok=True, parents=True)
        return v
    
    @validator("LOG_FILE")
    def validate_log_file(cls, v, values):
        """Ensure log directory exists if logging to file"""
        if values.get("LOG_TO_FILE"):
            log_path = Path(v)
            log_path.parent.mkdir(exist_ok=True, parents=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Configure logging
def setup_logging():
    """Configure logging based on environment settings"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    logging_config = {
        "level": log_level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    
    if settings.LOG_TO_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(exist_ok=True, parents=True)
        logging_config["filename"] = settings.LOG_FILE
    
    logging.basicConfig(**logging_config)
    
    # Log environment info
    logger = logging.getLogger("environment")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Using mock data: {settings.USE_MOCK_DATA}")
    
    if settings.ENVIRONMENT == Environment.TEST:
        logger.warning("Running in TEST mode - using limited or mock data")
    elif settings.ENVIRONMENT == Environment.PROD:
        logger.info("Running in PRODUCTION mode")

# Initialize logging when module is imported
setup_logging()
