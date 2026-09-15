"""
Database Engine and Session Management for SatQuery AI.
Configured for PostgreSQL persistence using SQLAlchemy 2.0 with connection pooling.
Fallback to SQLite for local development or testing environments if PostgreSQL driver/database is unavailable.
"""

import os
import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from database.base import Base

logger = logging.getLogger("satquery.database")

# Default PostgreSQL Connection String
DEFAULT_POSTGRES_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/satquery_db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_POSTGRES_URL)

# Check if psycopg2 or psycopg driver is installed when attempting PostgreSQL
is_postgres_url = DATABASE_URL.startswith("postgresql")
has_postgres_driver = False

if is_postgres_url:
    try:
        import psycopg2  # noqa: F401
        has_postgres_driver = True
    except ImportError:
        try:
            import psycopg  # noqa: F401
            # Adjust dialect for psycopg v3 if psycopg2 not found
            if "+psycopg2" in DATABASE_URL:
                DATABASE_URL = DATABASE_URL.replace("+psycopg2", "+psycopg")
            has_postgres_driver = True
        except ImportError:
            has_postgres_driver = False

if is_postgres_url and has_postgres_driver:
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
    except Exception as e:
        logger.warning(f"Failed to initialize PostgreSQL engine: {e}. Falling back to SQLite.")
        DATABASE_URL = "sqlite:///./satquery.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
else:
    if is_postgres_url and not has_postgres_driver:
        logger.warning("PostgreSQL URL configured but neither 'psycopg2' nor 'psycopg' module installed. Falling back to SQLite.")
    DATABASE_URL = "sqlite:///./satquery.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session per request.
    Automatically closes session upon request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> dict:
    """
    Checks database connection health and returns active stats.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            return {
                "status": "healthy" if result == 1 else "unhealthy",
                "database_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
                "dialect": engine.dialect.name
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "dialect": engine.dialect.name
        }


def init_db():
    """
    Initializes database tables by creating all registered SQLAlchemy models.
    Imports model modules to ensure metadata registration before create_all.
    """
    import database.models  # noqa: F401
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Successfully initialized database tables for dialect: {engine.dialect.name}")
    except Exception as e:
        logger.warning(f"Database initialization exception: {e}")
