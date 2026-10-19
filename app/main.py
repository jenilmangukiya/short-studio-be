from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from . import models  # noqa: F401  (register tables)
from .config import settings
from .db import Base, engine
from .routers import auth, channels, videos


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)  # swap for Alembic migrations before real data
    yield


app = FastAPI(title="Shorts Studio", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(channels.router)
app.include_router(videos.router)


@app.get("/health")
def health():
    return {"ok": True}
