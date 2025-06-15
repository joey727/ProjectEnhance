from sqlalchemy import TIMESTAMP, Column, ForeignKey, Integer, String, text, Boolean, JSON, Enum, Float
from sqlalchemy.orm import relationship
from backend.app.database import Base
import enum
from datetime import datetime


class EnhancementType(enum.Enum):
    CLARITY = "clarity"
    COLOR = "color"
    RESOLUTION = "resolution"


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=True)
    picture = Column(String, nullable=True)
    credits = Column(Integer, default=10)
    created_at = Column(TIMESTAMP(timezone=True),
                        nullable=False, server_default=text('now()'))
    last_login = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    images = relationship("Image", back_populates="owner",
                          cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey(
        'users.id', ondelete='CASCADE'), nullable=False)
    original_filename = Column(String, nullable=False)
    original_url = Column(String, nullable=False)
    enhanced_url = Column(String, nullable=True)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True),
                        nullable=False, server_default=text('now()'))

    # Relationships
    owner = relationship("User", back_populates="images")
    enhancement = relationship(
        "Enhancement", back_populates="image", uselist=False, cascade="all, delete-orphan")


class Enhancement(Base):
    __tablename__ = 'enhancements'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey(
        'images.id', ondelete='CASCADE'), nullable=False, unique=True)
    type = Column(Enum(EnhancementType), nullable=False)
    parameters = Column(JSON, nullable=True)
    # pending, processing, completed, failed
    status = Column(String, nullable=False, default='pending')
    result_url = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)  # Time taken in seconds
    created_at = Column(TIMESTAMP(timezone=True),
                        nullable=False, server_default=text('now()'))
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    image = relationship("Image", back_populates="enhancement")


class UsageLog(Base):
    __tablename__ = 'usage_logs'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey(
        'users.id', ondelete='CASCADE'), nullable=False)
    enhancement_id = Column(Integer, ForeignKey(
        'enhancements.id', ondelete='SET NULL'), nullable=True)
    credits_used = Column(Integer, nullable=False, default=1)
    created_at = Column(TIMESTAMP(timezone=True),
                        nullable=False, server_default=text('now()'))
