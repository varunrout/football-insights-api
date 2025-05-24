from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import dashboard_main
from app.routers import xt_analytics  # Import our new XT analytics router
from app.routers import tactical_metrics  # Import our new tactical metrics router

app = FastAPI(
 title="Football Insights API",
 description="API for accessing football analytics and insights",
 version="0.1.0"
)

# Configure CORS
app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],  # For development; restrict in production
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

# Include routers
app.include_router(dashboard_main.router)
app.include_router(xt_analytics.router)  # Add the XT analytics router
app.include_router(tactical_metrics.router)  # Add the tactical metrics router


@app.get("/")
async def root():
 return {
 "message": "Welcome to Football Insights API",
 "docs": "/docs",
 "version": "0.1.0"
 }
