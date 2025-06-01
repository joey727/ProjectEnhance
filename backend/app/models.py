from sqlalchemy import TIMESTAMP, Column, ForeignKey, Integer, String, text
from sqlalchemy.orm import relationship
from backend.app.database import Base


class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, nullable=False)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone='True'),
                        nullable=False, server_default=text('now()'))


class Image(Base):
    __tablename__ = 'images'

    image_id = Column(Integer, primary_key=True, nullable=False)
    owner_id = Column(Integer, ForeignKey(
        'users.user_id', ondelete='CASCADE'), nullable=False)
    image_type = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    upload_time = Column(TIMESTAMP(timezone='True'),
                         nullable=False, server_default=text('now()'))

    owner = relationship('User')


class Enhancement(Base):
    __tablename__ = 'enhancements'

    enhancement_id = Column(Integer, primary_key=True, nullable=False)
    image_id = Column(Integer, ForeignKey(
        'images.image_id', ondelete='CASCADE'), nullable=False)
    enhancement_type = Column(String, nullable=False)
    parameters = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone='True'),
                        nullable=False, server_default=text('now()'))


class Feedback(Base):
    __tablename__ = 'feedback'

    feedback_id = Column(Integer, primary_key=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.user_id', ondelete='SET NULL'))
    enhancement_id = Column(Integer, ForeignKey(
        'enhancements.enhancement_id', ondelete='CASCADE'), nullable=False)
    rating = Column(Integer, nullable=False)
    comment = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone='True'),
                        nullable=False, server_default=text('now()'))
