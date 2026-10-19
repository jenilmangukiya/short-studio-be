from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user, get_owned_channel
from ..models import Channel, User
from ..schemas import ChannelCreate, ChannelOut, ChannelUpdate

router = APIRouter(prefix="/channels", tags=["channels"])


@router.post("", response_model=ChannelOut, status_code=201)
def create_channel(body: ChannelCreate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ch = Channel(user_id=user.id, **body.model_dump())
    db.add(ch)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "You already have a channel with this name on this platform")
    return ch


@router.get("", response_model=list[ChannelOut])
def list_channels(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return db.scalars(select(Channel).where(Channel.user_id == user.id).order_by(Channel.created_at)).all()


@router.get("/{channel_id}", response_model=ChannelOut)
def get_channel(channel: Channel = Depends(get_owned_channel)):
    return channel


@router.patch("/{channel_id}", response_model=ChannelOut)
def update_channel(body: ChannelUpdate, channel: Channel = Depends(get_owned_channel), db: Session = Depends(get_db)):
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(channel, k, v)
    db.commit()
    return channel


@router.delete("/{channel_id}", status_code=204)
def delete_channel(channel: Channel = Depends(get_owned_channel), db: Session = Depends(get_db)):
    db.delete(channel)
    db.commit()
