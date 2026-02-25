import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://news:news@localhost:5432/news_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
