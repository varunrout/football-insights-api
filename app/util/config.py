from pydantic_settings import BaseSettings

class Settings(BaseSettings):
 APP_NAME: str = "Football Insights Lab API"
 ENV: str = "DEV"
 DATA_PATH: str = "./data"

 class Config:
     env_file = ".env"

settings = Settings()
