import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config.environment import settings, setup_logging
from app.api import dashboard, xt_analytics, player_comparison, tactical_insights, positional_analysis, matchup_analysis, player_analysis

# Set up logging
setup_logging()
logger = logging.getLogger("main")

# Create FastAPI app
app = FastAPI(
    title="Football Insights API",
    description="API for football analytics and visualizations",
    version="0.1.0"
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

# Include routers for different visualization categories
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(xt_analytics.router, prefix="/api/xt", tags=["xT Analytics"])
app.include_router(player_comparison.router, prefix="/api/player-comparison", tags=["Player Comparison"])
app.include_router(tactical_insights.router, prefix="/api/tactics", tags=["Tactical Insights"])
app.include_router(positional_analysis.router, prefix="/api/positions", tags=["Positional Analysis"])
app.include_router(matchup_analysis.router, prefix="/api/matchups", tags=["Match-Up Analysis"])
app.include_router(player_analysis.router, prefix="/api/players", tags=["Player Analysis"])

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Football Insights API",
        "version": "0.1.0",
        "documentation": "/docs",
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
