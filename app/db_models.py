from sqlalchemy import Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import mapped_column, relationship
from .database import Base
from datetime import datetime

class DeployedModel(Base):
    __tablename__ = "deployed_model"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(Integer, nullable=False)  # disrupt_model(1) 또는 detect_model(2)
    version = Column(Integer, nullable=False)
    deployment_time = Column(DateTime, default=datetime.utcnow)

class Performance(Base):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(Integer, nullable=False)  # disrupt_model(1) 또는 detect_model(2)
    version = Column(Integer, nullable=False)
    data_version = Column(Integer, nullable=False)
    bbox_loss = Column(Float, nullable=False)
    landmarks_loss = Column(Float, nullable=False)
    perturbation_loss = Column(Float, nullable=False)
    identity_loss = Column(Float, nullable=False)
    total_loss = Column(Float, nullable=False)
    evaluation_time = Column(DateTime, default=datetime.utcnow)

class PreprocessingEntity(Base):
    __tablename__ = "preprocessing"

    id = Column(Integer, primary_key=True, index=True)
    npz_url = Column(String, nullable=False)  # 널 불가 문자열
    is_tmp = Column(Boolean, default=True)    # 기본값 True인 불리언
    now_ver = Column(Integer, nullable=True)  # 널 가능 정수값