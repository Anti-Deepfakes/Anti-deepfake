from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import os

user = os.getenv("DB_USER")     # "root"
passwd = os.getenv("DB_PASSWD") # "1234"
host = os.getenv("DB_HOST")     # "127.0.0.1"
port = os.getenv("DB_PORT", "3306")     # "3306"
db = os.getenv("DB_NAME")       # "mydb"
print(f"DB_USER: {os.getenv('DB_USER')}")
print(f"DB_HOST: {os.getenv('DB_HOST')}")
print(f"DB_PORT: {os.getenv('DB_PORT')}")
DB_URL = f'mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}?charset=utf8'

engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False,autoflush=False, bind=engine)
Base = declarative_base()
Base.metadata.create_all(bind=engine)
# DB 세션을 반환하는 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()