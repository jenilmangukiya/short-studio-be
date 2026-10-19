import uuid

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import Channel, User, Video
from .security import decode_token

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")


def current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    try:
        uid = uuid.UUID(decode_token(token)["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(User, uid)
    if not user or not user.is_active:
        raise HTTPException(401, "User not found or inactive")
    return user


def get_owned_channel(channel_id: uuid.UUID, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Channel:
    ch = db.scalar(select(Channel).where(Channel.id == channel_id, Channel.user_id == user.id))
    if not ch:
        raise HTTPException(404, "Channel not found")
    return ch


def get_owned_video(video_id: uuid.UUID, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Video:
    video = db.scalar(
        select(Video).join(Channel).where(Video.id == video_id, Channel.user_id == user.id)
    )
    if not video:
        raise HTTPException(404, "Video not found")
    return video
