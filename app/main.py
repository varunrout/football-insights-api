from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.util.config import settings
from .routers import dashboard_main

app = FastAPI(
 title=settings.APP_NAME,
 version="1.0.0",
 description="Backend API for football analytics"
)

app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)


app.include_router(dashboard_main.router)  # Dashboard API


@app.get("/")
async def root():
 return {"message": "Welcome to Football Insights Lab API"}
