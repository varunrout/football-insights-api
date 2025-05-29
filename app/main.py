from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, List

# Import configuration
from app.config.environment import settings

# Import API routers
from app.api import dashboard, player_analysis, player_comparison, positional_analysis, xt_analytics, tactical_insights, matchup_analysis

# Setup logging
logger = logging.getLogger(__name__)

# Create the FastAPI application
app = FastAPI(
    title="Football Insights API",
    description="Advanced football analytics API built on StatsBomb data",
    version="0.1.0",
)

# Configure CORS
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict:
    """
    Root endpoint that returns API information
    """
    return {
        "name": "Football Insights API",
        "version": "0.1.0",
        "description": "Advanced football analytics API built on StatsBomb data",
        "documentation": "/docs",
    }

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict:
    """
    Health check endpoint
    """
    return {"status": "healthy"}

# Include routers with appropriate prefixes
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(player_analysis.router, prefix="/api/v1/player-analysis", tags=["Player Analysis"])
app.include_router(player_comparison.router, prefix="/api/v1/player-comparison", tags=["Player Comparison"])
app.include_router(positional_analysis.router, prefix="/api/v1/positional-analysis", tags=["Positional Analysis"])
app.include_router(xt_analytics.router, prefix="/api/v1/xt-analytics", tags=["xT Analytics"])
app.include_router(tactical_insights.router, prefix="/api/v1/tactical-insights", tags=["Tactical Insights"])
app.include_router(matchup_analysis.router, prefix="/api/v1/matchup-analysis", tags=["Matchup Analysis"])

# Event handlers
@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup
    """
    logger.info("Starting Football Insights API")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on application shutdown
    """
    logger.info("Shutting down Football Insights API")

# Run the API with Uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
