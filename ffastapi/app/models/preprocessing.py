from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PreprocessingEntity(Base):
    __tablename__ = "preprocessing"

    id = Column(Integer, primary_key=True, index=True)
    npz_url = Column()
    is_tmp = Column()
    now_ver = Column()