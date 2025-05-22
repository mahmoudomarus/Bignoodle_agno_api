from typing import Generator

from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from db.url import get_db_url

# Create SQLAlchemy Engine using a database URL with optimized connection pool settings
db_url: str = get_db_url()
db_engine: Engine = create_engine(
    db_url, 
    poolclass=QueuePool,
    pool_pre_ping=True,
    pool_size=5,  # Limit concurrent connections
    max_overflow=10,  # Allow some overflow during traffic spikes
    pool_timeout=30,  # Wait up to 30 seconds for a connection
    pool_recycle=1800,  # Recycle connections after 30 minutes
)

# Create a SessionLocal class
SessionLocal: sessionmaker[Session] = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get a database session.

    Yields:
        Session: An SQLAlchemy database session.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Function to check database connection pool status (for debugging)
def get_pool_status():
    """Get current database connection pool status"""
    return {
        "pool_size": db_engine.pool.size(),
        "checkedin": db_engine.pool.checkedin(),
        "checkedout": db_engine.pool.checkedout(),
        "overflow": db_engine.pool.overflow(),
    }
