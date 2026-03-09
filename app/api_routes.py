"""
FILE: backend/app/api_routes.py
FastAPI route definitions for all API endpoints
"""

import logging
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.attendance_service import attendance_service
from app.config import settings
from app.database import Company, Employee, UserAccount, get_db
from app.embedding_service import embedding_service
from app.face_matching import face_matching_service
from app.security import (
    create_access_token, verify_token, hash_password, verify_password,
    get_current_user
)

logger = logging.getLogger(__name__)

# ─── Routers ───────────────────────────────────────────────────────────────────

router_auth = APIRouter(prefix="/auth", tags=["Authentication"])
router_employee = APIRouter(prefix="/employees", tags=["Employees"])
router_attendance = APIRouter(prefix="/attendance", tags=["Attendance"])
router_admin = APIRouter(prefix="/admin", tags=["Admin"])

bearer_scheme = HTTPBearer()


# ─── Pydantic Schemas ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_id: Optional[str] = None


class RegisterEmployeeRequest(BaseModel):
    company_id: str
    employee_code: str
    name: str
    department: Optional[str] = None
    position: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class VerifyFaceRequest(BaseModel):
    company_id: str
    device_id: Optional[str] = None


# ─── Auth Routes ───────────────────────────────────────────────────────────────

@router_auth.post("/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate an HR admin user and return JWT tokens."""
    result = await db.execute(
        select(UserAccount).where(UserAccount.email == request.email, UserAccount.is_active == True)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(
        data={"sub": str(user.user_id), "company_id": str(user.company_id), "role": user.role}
    )

    user.last_login = datetime.now(timezone.utc)
    await db.flush()

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "user_id": str(user.user_id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "company_id": str(user.company_id),
        }
    }


@router_auth.post("/refresh")
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db)
):
    """Issue a new access token from a valid existing token."""
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    new_token = create_access_token(data=payload)
    return {"access_token": new_token, "token_type": "bearer"}


# ─── Employee Routes ───────────────────────────────────────────────────────────

@router_employee.post("/register")
async def register_employee(
    company_id: str = Form(...),
    employee_code: str = Form(...),
    name: str = Form(...),
    department: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    face_image: UploadFile = File(..., description="Front-facing photo of employee"),
    device_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    POST /employees/register
    Register a new employee with their face embedding.
    Requires HR admin authentication.
    """
    # Authorization: user must belong to same company
    if current_user["company_id"] != company_id and current_user["role"] != "super_admin":
        raise HTTPException(status_code=403, detail="Not authorized to register employees for this company")

    # Validate image
    if face_image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Image must be JPEG, PNG, or WebP")

    image_bytes = await face_image.read()
    if len(image_bytes) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image too large (max {settings.MAX_IMAGE_SIZE_MB}MB)")

    # Extract face embedding
    extraction = await embedding_service.extract_embedding(image_bytes)
    if not extraction["success"]:
        raise HTTPException(
            status_code=422,
            detail=f"Face registration failed: {extraction.get('reason', 'unknown')}"
        )

    embedding = extraction["embedding"]

    # Check for duplicate employee code
    existing = await db.execute(
        select(Employee).where(
            Employee.company_id == company_id,
            Employee.employee_code == employee_code,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Employee code '{employee_code}' already exists")

    # Save profile photo
    photo_filename = f"{uuid.uuid4()}.jpg"
    photo_path = os.path.join(settings.PROFILE_PHOTOS_DIR, photo_filename)
    os.makedirs(settings.PROFILE_PHOTOS_DIR, exist_ok=True)
    with open(photo_path, "wb") as f:
        f.write(image_bytes)

    # Create employee record
    new_employee = Employee(
        company_id=company_id,
        employee_code=employee_code,
        name=name,
        department=department,
        position=position,
        email=email,
        phone=phone,
        profile_photo=photo_path,
        face_embedding=embedding.tolist(),
        face_registered_at=datetime.now(timezone.utc),
        registered_device_id=device_id,
    )
    db.add(new_employee)
    await db.flush()

    # Add to FAISS index
    faiss_id = await embedding_service.add_to_index(str(new_employee.employee_id), embedding, db)
    new_employee.faiss_index_id = faiss_id
    await db.flush()

    logger.info(f"Employee registered: {name} ({new_employee.employee_id}) company={company_id}")

    return {
        "success": True,
        "employee_id": str(new_employee.employee_id),
        "employee_code": employee_code,
        "name": name,
        "department": department,
        "faiss_index_id": faiss_id,
        "face_registered": True,
        "det_score": extraction.get("det_score"),
        "message": f"Employee '{name}' registered successfully",
    }


@router_employee.get("/{employee_id}")
async def get_employee(
    employee_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """GET /employees/{employee_id} - Retrieve employee details."""
    result = await db.execute(
        select(Employee).where(
            Employee.employee_id == employee_id,
            Employee.company_id == current_user["company_id"],
        )
    )
    emp = result.scalar_one_or_none()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    return {
        "employee_id": str(emp.employee_id),
        "employee_code": emp.employee_code,
        "name": emp.name,
        "department": emp.department,
        "position": emp.position,
        "email": emp.email,
        "phone": emp.phone,
        "face_registered": emp.face_embedding is not None,
        "face_registered_at": emp.face_registered_at.isoformat() if emp.face_registered_at else None,
        "is_active": emp.is_active,
        "created_at": emp.created_at.isoformat() if emp.created_at else None,
    }


@router_employee.get("/")
async def list_employees(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, le=200),
    department: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """GET /employees - List all employees for a company."""
    from sqlalchemy import func

    query = select(Employee).where(
        Employee.company_id == current_user["company_id"],
        Employee.is_active == True,
    )
    if department:
        query = query.where(Employee.department == department)

    result = await db.execute(query.offset((page - 1) * page_size).limit(page_size))
    employees = result.scalars().all()

    return {
        "page": page,
        "page_size": page_size,
        "employees": [
            {
                "employee_id": str(e.employee_id),
                "employee_code": e.employee_code,
                "name": e.name,
                "department": e.department,
                "position": e.position,
                "face_registered": e.face_embedding is not None,
            }
            for e in employees
        ]
    }


# ─── Face Verification Routes ──────────────────────────────────────────────────

@router_attendance.post("/verify_face")
async def verify_face(
    company_id: str = Form(...),
    device_id: Optional[str] = Form(None),
    face_image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    POST /attendance/verify_face
    Verify a face image (no attendance recording — just identification).
    Used for testing and admin purposes.
    """
    image_bytes = await face_image.read()
    if len(image_bytes) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large")

    match = await face_matching_service.verify_face(image_bytes, company_id, db, device_id)

    return {
        "matched": match.matched,
        "employee_id": match.employee_id,
        "employee_name": match.employee_name,
        "department": match.department,
        "similarity_score": round(match.similarity_score, 4),
        "liveness_score": round(match.liveness_score, 4),
        "det_score": round(match.det_score, 4),
        "reason": match.reason,
        "processing_time_ms": round(match.processing_time_ms, 1),
    }


@router_attendance.post("/clock_in")
async def clock_in(
    company_id: str = Form(...),
    device_id: Optional[str] = Form(None),
    gps_lat: Optional[float] = Form(None),
    gps_lng: Optional[float] = Form(None),
    gps_accuracy: Optional[float] = Form(None),
    face_image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    POST /attendance/clock_in
    Verify face and record clock-in.
    """
    image_bytes = await face_image.read()
    if len(image_bytes) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large")

    # Build GPS dict if coordinates provided
    gps_location = None
    if gps_lat is not None and gps_lng is not None:
        gps_location = {"lat": gps_lat, "lng": gps_lng, "accuracy": gps_accuracy}

    # Verify face
    match = await face_matching_service.verify_face(image_bytes, company_id, db, device_id)

    # Record attendance
    result = await attendance_service.clock_in(
        match_result=match,
        company_id=company_id,
        db=db,
        gps_location=gps_location,
        device_id=device_id,
    )

    return result


@router_attendance.post("/clock_out")
async def clock_out(
    company_id: str = Form(...),
    device_id: Optional[str] = Form(None),
    gps_lat: Optional[float] = Form(None),
    gps_lng: Optional[float] = Form(None),
    gps_accuracy: Optional[float] = Form(None),
    face_image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    POST /attendance/clock_out
    Verify face and record clock-out.
    """
    image_bytes = await face_image.read()

    gps_location = None
    if gps_lat is not None and gps_lng is not None:
        gps_location = {"lat": gps_lat, "lng": gps_lng, "accuracy": gps_accuracy}

    match = await face_matching_service.verify_face(image_bytes, company_id, db, device_id)

    result = await attendance_service.clock_out(
        match_result=match,
        company_id=company_id,
        db=db,
        gps_location=gps_location,
        device_id=device_id,
    )

    return result


# ─── Attendance History Routes ─────────────────────────────────────────────────

@router_attendance.get("/history")
async def attendance_history(
    employee_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    GET /attendance/history
    Fetch paginated attendance records for a company.
    Employees can only see their own records; HR admins can see all.
    """
    company_id = current_user["company_id"]

    # Employees can only query their own records
    if current_user["role"] == "employee" and employee_id != current_user["employee_id"]:
        employee_id = current_user["employee_id"]

    result = await attendance_service.get_attendance_history(
        employee_id=employee_id,
        company_id=company_id,
        db=db,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
    )
    return result


@router_attendance.get("/summary/{target_date}")
async def attendance_summary(
    target_date: date,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """GET /attendance/summary/{date} - Daily attendance summary."""
    return await attendance_service.get_daily_summary(
        company_id=current_user["company_id"],
        target_date=target_date,
        db=db,
    )


# ─── Admin Routes ──────────────────────────────────────────────────────────────

@router_admin.post("/rebuild_index")
async def rebuild_faiss_index(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """POST /admin/rebuild_index - Rebuild FAISS index from database."""
    if current_user["role"] not in ["super_admin", "hr_admin"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    count = await embedding_service.build_index(db)
    return {
        "success": True,
        "employees_indexed": count,
        "message": f"FAISS index rebuilt with {count} employee embeddings",
    }


@router_admin.get("/system_status")
async def system_status(current_user: dict = Depends(get_current_user)):
    """GET /admin/system_status - Health check with system info."""
    return {
        "status": "healthy",
        "faiss_index_size": embedding_service.index_size,
        "model_ready": True,
        "embedding_size": settings.EMBEDDING_SIZE,
        "match_threshold": settings.FACE_MATCH_THRESHOLD,
        "liveness_enabled": settings.LIVENESS_ENABLED,
    }
