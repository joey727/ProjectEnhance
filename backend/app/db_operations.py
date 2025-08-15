# from sqlalchemy.orm import Session
# from sqlalchemy.exc import IntegrityError
# from sqlalchemy import func, case
# from typing import List, Optional, Dict, Any, Tuple
# from .models import User, Image, Enhancement, UsageLog, EnhancementType
# from datetime import datetime
# from contextlib import contextmanager


# @contextmanager
# def transaction(db: Session):
#     """Context manager for database transactions."""
#     try:
#         yield
#         db.commit()
#     except Exception as e:
#         db.rollback()
#         raise e


# # User Operations
# def get_or_create_user(db: Session, email: str, name: Optional[str] = None, picture: Optional[str] = None) -> Tuple[User, bool]:
#     """Get existing user or create new one. Returns (user, is_new)."""
#     user = db.query(User).filter(User.email == email).first()
#     if user:
#         user.last_login = datetime.now()
#         db.commit()
#         return user, False

#     user = User(
#         email=email,
#         name=name,
#         picture=picture,
#         credits=10,
#         last_login=datetime.now()
#     )
#     db.add(user)
#     db.commit()
#     db.refresh(user)
#     return user, True


# def update_user_credits(db: Session, user_id: int, credits: int) -> Optional[User]:
#     """Update user credits with transaction management."""
#     with transaction(db):
#         user = db.query(User).filter(User.id == user_id).first()
#         if user:
#             user.credits = credits
#             db.refresh(user)
#         return user


# # Image Operations
# def create_image_with_enhancement(
#     db: Session,
#     owner_id: int,
#     original_filename: str,
#     original_url: str,
#     file_size: int,
#     width: Optional[int] = None,
#     height: Optional[int] = None,
#     enhancement_type: EnhancementType = EnhancementType.CLARITY,
#     parameters: Optional[Dict[str, Any]] = None
# ) -> Tuple[Image, Enhancement]:
#     """Create a new image with its enhancement request in a single transaction."""
#     with transaction(db):
#         # Create image
#         image = Image(
#             owner_id=owner_id,
#             original_filename=original_filename,
#             original_url=original_url,
#             file_size=file_size,
#             width=width,
#             height=height
#         )
#         db.add(image)
#         db.flush()  # Get the image ID without committing

#         # Create enhancement
#         enhancement = Enhancement(
#             image_id=image.id,
#             type=enhancement_type,
#             parameters=parameters,
#             status='pending'
#         )
#         db.add(enhancement)
#         db.flush()

#         return image, enhancement


# def get_user_images_with_enhancements(
#     db: Session,
#     user_id: int,
#     skip: int = 0,
#     limit: int = 10
# ) -> List[Tuple[Image, Enhancement]]:
#     """Get paginated list of user's images with their enhancement status."""
#     return db.query(Image, Enhancement)\
#         .join(Enhancement)\
#         .filter(Image.owner_id == user_id)\
#         .order_by(Image.created_at.desc())\
#         .offset(skip)\
#         .limit(limit)\
#         .all()


# def get_image_with_enhancement(db: Session, image_id: int) -> Optional[Tuple[Image, Enhancement]]:
#     """Get image with its enhancement by ID."""
#     return db.query(Image, Enhancement)\
#         .join(Enhancement)\
#         .filter(Image.id == image_id)\
#         .first()


# # Enhancement Operations
# def update_enhancement_status(
#     db: Session,
#     enhancement_id: int,
#     status: str,
#     result_url: Optional[str] = None,
#     error_message: Optional[str] = None,
#     processing_time: Optional[float] = None
# ) -> Optional[Enhancement]:
#     """Update enhancement status with additional metadata."""
#     with transaction(db):
#         enhancement = db.query(Enhancement).filter(
#             Enhancement.id == enhancement_id).first()
#         if enhancement:
#             enhancement.status = status
#             if result_url:
#                 enhancement.result_url = result_url
#             if error_message:
#                 enhancement.error_message = error_message
#             if processing_time:
#                 enhancement.processing_time = processing_time
#             if status in ['completed', 'failed']:
#                 enhancement.completed_at = datetime.now()
#             db.refresh(enhancement)
#         return enhancement


# def get_pending_enhancements(db: Session, limit: int = 10) -> List[Enhancement]:
#     """Get list of pending enhancements for processing."""
#     return db.query(Enhancement)\
#         .filter(Enhancement.status == 'pending')\
#         .order_by(Enhancement.created_at.asc())\
#         .limit(limit)\
#         .all()


# def get_user_enhancement_stats(db: Session, user_id: int) -> Dict[str, Any]:
#     """Get enhancement statistics for a user."""
#     stats = db.query(
#         func.count(Enhancement.id).label('total'),
#         func.sum(case((Enhancement.status == 'completed', 1), else_=0)
#                  ).label('completed'),
#         func.sum(case((Enhancement.status == 'failed', 1), else_=0)
#                  ).label('failed'),
#         func.avg(Enhancement.processing_time).label('avg_processing_time')
#     ).join(Image).filter(Image.owner_id == user_id).first()

#     return {
#         'total': stats.total or 0,
#         'completed': stats.completed or 0,
#         'failed': stats.failed or 0,
#         'avg_processing_time': float(stats.avg_processing_time) if stats.avg_processing_time else None
#     }


# # Usage Log Operations
# def create_usage_log(
#     db: Session,
#     user_id: int,
#     enhancement_id: Optional[int] = None,
#     credits_used: int = 1
# ) -> UsageLog:
#     """Create a new usage log entry."""
#     usage_log = UsageLog(
#         user_id=user_id,
#         enhancement_id=enhancement_id,
#         credits_used=credits_used
#     )
#     db.add(usage_log)
#     db.commit()
#     db.refresh(usage_log)
#     return usage_log


# def get_user_usage_logs(
#     db: Session,
#     user_id: int,
#     skip: int = 0,
#     limit: int = 10
# ) -> List[UsageLog]:
#     """Get paginated list of user's usage logs."""
#     return db.query(UsageLog)\
#         .filter(UsageLog.user_id == user_id)\
#         .order_by(UsageLog.created_at.desc())\
#         .offset(skip)\
#         .limit(limit)\
#         .all()


# def get_total_credits_used(db: Session, user_id: int) -> int:
#     """Get total credits used by a user."""
#     result = db.query(UsageLog)\
#         .filter(UsageLog.user_id == user_id)\
#         .with_entities(func.sum(UsageLog.credits_used))\
#         .scalar()
#     return result or 0
