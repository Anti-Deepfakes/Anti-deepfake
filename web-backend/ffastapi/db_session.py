from sqlalchemy.orm import Session
from contextlib import contextmanager

# DB 세션을 반환하는 함수
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()