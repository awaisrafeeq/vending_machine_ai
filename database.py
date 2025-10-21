import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base  # 2.0 style
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Configure it in your environment.")

# If your Render URL doesn't already include ?sslmode=require (most do), uncomment:
# if "sslmode=" not in DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")):
#     DATABASE_URL += ("&"
#  if "?" in DATABASE_URL else "?") + "sslmode=require"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # avoids stale connection errors
    pool_recycle=1800,       # optional: recycle every 30 min
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_connection() -> bool:
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1")).scalar() == 1
