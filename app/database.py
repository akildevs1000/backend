"""
FILE: backend/app/database.py
Database models and connection management using SQLAlchemy async
Supports both PostgreSQL (production) and SQLite (local dev)
"""

import uuid
import json
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    Column, String, DateTime, Float, Text, Boolean,
    ForeignKey, Index, JSON, Integer, LargeBinary, TypeDecorator, func
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from app.config import settings


# ─── Custom Type for embedding (JSON list for SQLite, ARRAY for PG) ──────────

class EmbeddingArray(TypeDecorator):
    """Stores float arrays as JSON text — works with both SQLite and PostgreSQL."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value if isinstance(value, list) else list(value))
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


# ─── Engine & Session ──────────────────────────────────────────────────────────

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

engine_kwargs = {
    "echo": settings.DEBUG,
}
if not _is_sqlite:
    engine_kwargs.update({
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_pre_ping": True,
    })

engine = create_async_engine(settings.DATABASE_URL, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


# ─── Database Dependency ───────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─── Models ────────────────────────────────────────────────────────────────────

class Company(Base):
    __tablename__ = "companies"

    company_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_name = Column(String(255), nullable=False, unique=True)
    company_code = Column(String(50), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    is_active = Column(Boolean, default=True)
    subscription_plan = Column(String(50), default="standard")
    max_employees = Column(Integer, default=1000)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    employees = relationship("Employee", back_populates="company", cascade="all, delete-orphan")
    admins = Column(JSON, default=list)

    __table_args__ = (
        Index("idx_company_code", "company_code"),
    )


class Employee(Base):
    __tablename__ = "employees"

    employee_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String(36), ForeignKey("companies.company_id", ondelete="CASCADE"), nullable=False)
    employee_code = Column(String(100), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    department = Column(String(100), nullable=True)
    position = Column(String(100), nullable=True)
    profile_photo = Column(String(512), nullable=True)

    # Face Recognition
    face_embedding = Column(EmbeddingArray, nullable=True)  # 512-dim ArcFace embedding as JSON
    embedding_version = Column(String(20), default="arcface_r100")
    face_registered_at = Column(DateTime, nullable=True)
    faiss_index_id = Column(Integer, nullable=True)

    # Device Binding
    registered_device_id = Column(String(255), nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    company = relationship("Company", back_populates="employees")
    attendance_records = relationship("Attendance", back_populates="employee", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_employee_company", "company_id"),
        Index("idx_employee_code", "employee_code", "company_id"),
        Index("idx_faiss_index_id", "faiss_index_id"),
    )


class Attendance(Base):
    __tablename__ = "attendance"

    attendance_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(36), ForeignKey("employees.employee_id", ondelete="CASCADE"), nullable=False)
    company_id = Column(String(36), ForeignKey("companies.company_id", ondelete="CASCADE"), nullable=False)

    # Timing
    date = Column(DateTime, nullable=False)
    clock_in = Column(DateTime, nullable=True)
    clock_out = Column(DateTime, nullable=True)
    work_duration_minutes = Column(Integer, nullable=True)

    # Location
    gps_location = Column(JSON, nullable=True)
    location_name = Column(String(255), nullable=True)

    # Verification
    verification_method = Column(String(50), default="face_recognition")
    similarity_score = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    device_id = Column(String(255), nullable=True)

    # Status
    status = Column(String(50), default="present")
    notes = Column(Text, nullable=True)

    # Metadata
    clock_in_photo = Column(String(512), nullable=True)
    clock_out_photo = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    employee = relationship("Employee", back_populates="attendance_records")

    __table_args__ = (
        Index("idx_attendance_employee_date", "employee_id", "date"),
        Index("idx_attendance_company_date", "company_id", "date"),
    )


class UserAccount(Base):
    """Admin / HR user accounts for managing the system"""
    __tablename__ = "user_accounts"

    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String(36), ForeignKey("companies.company_id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="hr_admin")
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_user_email", "email"),
    )


class AuditLog(Base):
    """Security and activity audit trail"""
    __tablename__ = "audit_logs"

    log_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String(36), nullable=True)
    user_id = Column(String(36), nullable=True)
    employee_id = Column(String(36), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(50), nullable=True)
    device_id = Column(String(255), nullable=True)
    success = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_audit_company_action", "company_id", "action"),
        Index("idx_audit_created_at", "created_at"),
    )


# ─── Database Initialization ───────────────────────────────────────────────────

async def init_db():
    """Create all tables on startup"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def seed_demo_data():
    """Insert demo company and admin user if not exists."""
    from app.security import hash_password

    async with AsyncSessionLocal() as session:
        # Check if demo company exists
        from sqlalchemy import select
        result = await session.execute(
            select(Company).where(Company.company_code == "ACME")
        )
        if result.scalar_one_or_none():
            return  # Already seeded

        company = Company(
            company_id="00000000-0000-0000-0000-000000000001",
            company_name="Acme Corporation",
            company_code="ACME",
            email="admin@acme.com",
            subscription_plan="enterprise",
            max_employees=100000,
        )
        session.add(company)

        admin = UserAccount(
            company_id="00000000-0000-0000-0000-000000000001",
            email="admin@acme.com",
            hashed_password=hash_password("Admin@123"),
            full_name="System Administrator",
            role="super_admin",
        )
        session.add(admin)
        await session.commit()


async def drop_db():
    """Drop all tables (use with caution)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
