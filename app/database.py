from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class ScanHistory(Base):
    __tablename__ = 'scan_history'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scanned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    qr_found: Mapped[bool] = mapped_column(Boolean, nullable=False)
    qr_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    wb_above_qr: Mapped[bool] = mapped_column(Boolean, nullable=False)
    code_below_qr: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)


_engine = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine():
    global _engine
    if _engine is None and settings.database_url:
        _engine = create_async_engine(settings.database_url, echo=False)
    return _engine


def get_session_factory() -> Optional[async_sessionmaker[AsyncSession]]:
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        if engine:
            _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


async def init_db() -> None:
    engine = get_engine()
    if engine is None:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_scan(
    qr_found: bool,
    qr_content: Optional[str],
    wb_above_qr: bool,
    code_below_qr: Optional[str],
    confidence: float,
) -> None:
    factory = get_session_factory()
    if factory is None:
        return
    async with factory() as session:
        record = ScanHistory(
            qr_found=qr_found,
            qr_content=qr_content,
            wb_above_qr=wb_above_qr,
            code_below_qr=code_below_qr,
            confidence=confidence,
        )
        session.add(record)
        await session.commit()


async def get_history(limit: int = 50) -> list[ScanHistory]:
    factory = get_session_factory()
    if factory is None:
        return []
    async with factory() as session:
        result = await session.execute(
            select(ScanHistory).order_by(ScanHistory.scanned_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
